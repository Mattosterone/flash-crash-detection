"""
pipeline_runner.py — Sequential pipeline runner with memory cleanup.

Runs each phase in order, freeing memory between steps.
Respects config.LIGHTWEIGHT_MODE and config.USE_FLOAT32.

Usage:
    python -m src.pipeline_runner
"""

import gc
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import setup_logging, set_reproducibility, save_result, load_result

logger = setup_logging(__name__)


def _log_mode() -> None:
    """Log active runtime configuration."""
    if config.LIGHTWEIGHT_MODE:
        logger.info(
            "Runtime: LIGHTWEIGHT_MODE=True (last %d rows), USE_FLOAT32=%s",
            config.SAMPLE_ROWS,
            config.USE_FLOAT32,
        )
    else:
        logger.info(
            "Runtime: LIGHTWEIGHT_MODE=False (full dataset), USE_FLOAT32=%s",
            config.USE_FLOAT32,
        )


def run_phase_a() -> None:
    """Phase A: Load raw data, clean, and save df_clean.parquet."""
    from src.data_prep import load_raw_data, clean_data

    logger.info("=== Phase A: Data loading & cleaning ===")
    df = load_raw_data()
    df, report = clean_data(df)
    logger.info("Cleaning report: %s", report)
    save_result(df, "df_clean")
    del df, report
    gc.collect()
    logger.info("Phase A complete — df_clean.parquet saved, memory freed")


def run_phase_b() -> None:
    """Phase B: CUSUM event filter → cusum_events.parquet."""
    import pandas as pd
    from src.cusum import compute_ewm_volatility, cusum_filter

    logger.info("=== Phase B: CUSUM event filter ===")
    df = load_result("df_clean")
    log_returns = df["log_return"].copy()
    del df
    gc.collect()

    vol = compute_ewm_volatility(log_returns, span=config.CUSUM_LOOKBACK)
    threshold = vol * config.CUSUM_THRESHOLD_MULTIPLIER
    events = cusum_filter(log_returns, threshold)

    del log_returns, vol, threshold
    gc.collect()

    # Save as single-column DataFrame
    events_df = pd.DataFrame({"event": 1}, index=events)
    save_result(events_df, "cusum_events")
    del events_df, events
    gc.collect()
    logger.info("Phase B complete — cusum_events.parquet saved, memory freed")


def run_phase_c() -> None:
    """Phase C: Triple Barrier labeling (standard + adaptive) → parquet."""
    from src.cusum import compute_ewm_volatility
    from src.labeling import apply_standard_tbm, apply_adaptive_tbm, compare_labeling_schemes
    from src.sample_weights import compute_sample_weights

    logger.info("=== Phase C: Triple Barrier labeling ===")
    df = load_result("df_clean")
    events_df = load_result("cusum_events")
    events = events_df.index
    del events_df
    gc.collect()

    close = df["close"].copy()
    log_returns = df["log_return"].copy()
    del df
    gc.collect()

    volatility = compute_ewm_volatility(log_returns, span=config.TBM_VOLATILITY_SPAN)
    del log_returns
    gc.collect()

    # --- Standard TBM ---
    logger.info(
        "Standard TBM: PT=%.1f, SL=%.1f, horizon=%d bars",
        config.TBM_PROFIT_TAKING, config.TBM_STOP_LOSS, config.TBM_HORIZON_BARS,
    )
    labels_standard = apply_standard_tbm(
        close=close,
        events=events,
        volatility=volatility,
        pt=config.TBM_PROFIT_TAKING,
        sl=config.TBM_STOP_LOSS,
        max_horizon=config.TBM_HORIZON_BARS,
    )
    gc.collect()

    # Sample weights for standard labels
    weights_standard = compute_sample_weights(
        event_starts=labels_standard.index,
        event_ends=labels_standard["t1"],
        close_index=close.index,
    )
    labels_standard["weight"] = weights_standard
    del weights_standard
    gc.collect()

    save_result(labels_standard, "labels_standard")
    gc.collect()

    # --- Adaptive TBM ---
    high_factor = config.TBM_ADAPTIVE_PT_HIGH_VOL / config.TBM_PROFIT_TAKING
    low_factor = config.TBM_ADAPTIVE_PT_LOW_VOL / config.TBM_PROFIT_TAKING
    logger.info(
        "Adaptive TBM: regime_window=%d, high_factor=%.3f, low_factor=%.3f",
        config.TBM_ADAPTIVE_VOL_LOOKBACK, high_factor, low_factor,
    )
    labels_adaptive = apply_adaptive_tbm(
        close=close,
        events=events,
        volatility=volatility,
        pt=config.TBM_PROFIT_TAKING,
        sl=config.TBM_STOP_LOSS,
        max_horizon=config.TBM_HORIZON_BARS,
        regime_window=config.TBM_ADAPTIVE_VOL_LOOKBACK,
        high_vol_pct=config.TBM_ADAPTIVE_VOL_PERCENTILE * 100.0,
        low_vol_pct=(1.0 - config.TBM_ADAPTIVE_VOL_PERCENTILE) * 100.0,
        high_vol_factor=high_factor,
        low_vol_factor=low_factor,
    )
    gc.collect()

    # Sample weights for adaptive labels
    weights_adaptive = compute_sample_weights(
        event_starts=labels_adaptive.index,
        event_ends=labels_adaptive["t1"],
        close_index=close.index,
    )
    labels_adaptive["weight"] = weights_adaptive
    del weights_adaptive, close, volatility, events
    gc.collect()

    save_result(labels_adaptive, "labels_adaptive")

    # --- Comparison ---
    comparison = compare_labeling_schemes(labels_standard, labels_adaptive)
    save_result(comparison.reset_index(), "labeling_comparison")

    del labels_standard, labels_adaptive, comparison
    gc.collect()
    logger.info("Phase C complete — labels saved, memory freed")


def main() -> None:
    """Run all pipeline phases sequentially."""
    set_reproducibility()
    _log_mode()

    logger.info("=" * 60)
    logger.info("Pipeline start")
    logger.info("=" * 60)

    run_phase_a()
    run_phase_b()
    run_phase_c()

    logger.info("=" * 60)
    logger.info("Pipeline complete. All outputs in data/processed/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
