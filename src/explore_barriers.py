"""
explore_barriers.py — Barrier setting exploration (no model training).

Step 1: Re-runs CUSUM with the current config multiplier (m=3.0) on
        df_clean.parquet and saves new cusum_events.parquet.

Step 2: Tests 14 barrier combinations with apply_standard_tbm(), recording
        crash rate, label horizons, timeout %, and mean absolute return at
        event time for each.

Step 3: Prints results sorted by crash_rate ascending and saves to
        results/tables/barrier_exploration.csv.

Usage
-----
    /opt/anaconda3/bin/python src/explore_barriers.py
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.cusum import compute_ewm_volatility, cusum_filter
from src.labeling import apply_standard_tbm
from src.utils import setup_logging

logger = setup_logging(__name__)

# ── barrier settings to explore ──────────────────────────────────────────
SETTINGS = [
    # (name,           pt,   sl,   horizon_bars)
    ("narrow_sym",     1.0,  1.0,   60),
    ("narrow_asym",    1.5,  1.0,   60),   # original paper setting
    ("medium_sym",     2.0,  2.0,   60),
    ("medium_asym",    2.0,  1.5,   60),
    ("wide_sym",       3.0,  3.0,   60),
    ("wide_asym",      3.0,  2.0,   60),
    ("crash_hard",     1.5,  2.5,   60),   # SL wider = crash harder to hit
    ("narrow_long",    1.0,  1.0,  120),
    ("medium_long",    2.0,  2.0,  120),
    ("wide_long",      3.0,  3.0,  120),
    ("medium_vlong",   2.0,  2.0,  240),
    ("wide_vlong",     3.0,  3.0,  240),
    ("extreme_wide",   4.0,  4.0,   60),
    ("extreme_long",   2.0,  2.0,  480),   # 8 hours
]


# ======================================================================
# STEP 1: RE-RUN CUSUM WITH m=3.0
# ======================================================================

def rerun_cusum(df_clean: pd.DataFrame) -> pd.DatetimeIndex:
    """Run CUSUM filter on df_clean using config.CUSUM_THRESHOLD_MULTIPLIER.

    Saves result to data/processed/cusum_events.parquet.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned OHLCV DataFrame with log_return column.

    Returns
    -------
    pd.DatetimeIndex
        New CUSUM event timestamps.
    """
    logger.info(
        "Running CUSUM with multiplier=%.1f on %d bars...",
        config.CUSUM_THRESHOLD_MULTIPLIER,
        len(df_clean),
    )

    volatility = compute_ewm_volatility(df_clean["log_return"], span=config.CUSUM_LOOKBACK)
    threshold = config.CUSUM_THRESHOLD_MULTIPLIER * volatility
    events = cusum_filter(df_clean["log_return"], threshold)

    event_rate = 100.0 * len(events) / len(df_clean)
    print(f"\nCUSUM m={config.CUSUM_THRESHOLD_MULTIPLIER}: "
          f"{len(events):,} events / {len(df_clean):,} bars "
          f"= {event_rate:.2f}% event rate\n")

    # Save
    out_path = config.PROCESSED_DATA_DIR / "cusum_events.parquet"
    pd.DataFrame(index=events).to_parquet(out_path)
    logger.info("Saved cusum_events.parquet -> %s  (%d events)", out_path, len(events))

    return events


# ======================================================================
# STEP 2: EXPLORE BARRIERS
# ======================================================================

def explore_barriers(
    df_clean: pd.DataFrame,
    events: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Run apply_standard_tbm for each setting and collect stats.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Full cleaned OHLCV DataFrame with log_return.
    events : pd.DatetimeIndex
        CUSUM event timestamps.

    Returns
    -------
    pd.DataFrame
        One row per setting with columns:
        setting, pt, sl, horizon_bars, n_events_labeled,
        crash_count, crash_rate_pct, no_crash_count,
        mean_horizon_min, median_horizon_min, max_horizon_min,
        pct_timeout, mean_abs_return_at_event.
    """
    # Volatility used for barrier widths (same EWM as labeling.py)
    volatility = (
        df_clean["log_return"]
        .ewm(span=config.VOLATILITY_EWM_SPAN, min_periods=config.VOLATILITY_EWM_SPAN)
        .std()
    )

    # Mean |log_return| at event times (computed once, reused for all settings)
    event_returns = df_clean["log_return"].reindex(events).abs()

    rows = []

    for name, pt, sl, horizon in SETTINGS:
        logger.info("Testing: %-20s pt=%.1f sl=%.1f horizon=%d", name, pt, sl, horizon)

        labels = apply_standard_tbm(
            close=df_clean["close"],
            events=events,
            volatility=volatility,
            pt=pt,
            sl=sl,
            max_horizon=horizon,
        )

        n = len(labels)
        if n == 0:
            logger.warning("Setting %s: 0 events labeled — skipping.", name)
            continue

        crash_count = int(labels["bin"].sum())
        no_crash_count = n - crash_count
        crash_rate = 100.0 * crash_count / n

        # Label horizon in minutes
        horizons_min = (labels["t1"] - labels.index).dt.total_seconds() / 60.0
        mean_h = horizons_min.mean()
        median_h = horizons_min.median()
        max_h = horizons_min.max()

        # Timeout %
        pct_timeout = 100.0 * (labels["barrier_type"] == "timeout").sum() / n

        # Mean |log_return| at labeled event times
        mean_ret = event_returns.reindex(labels.index).mean()

        rows.append({
            "setting":               name,
            "pt":                    pt,
            "sl":                    sl,
            "horizon_bars":          horizon,
            "n_events_labeled":      n,
            "crash_count":           crash_count,
            "crash_rate_pct":        round(crash_rate, 2),
            "no_crash_count":        no_crash_count,
            "mean_horizon_min":      round(mean_h, 1),
            "median_horizon_min":    round(median_h, 1),
            "max_horizon_min":       round(max_h, 1),
            "pct_timeout":           round(pct_timeout, 2),
            "mean_abs_return_at_event": round(float(mean_ret), 6),
        })

        del labels
        gc.collect()

    return pd.DataFrame(rows)


# ======================================================================
# STEP 3: PRINT + SAVE
# ======================================================================

def print_results(df: pd.DataFrame) -> None:
    """Print sorted table and recommendation summary."""
    sorted_df = df.sort_values("crash_rate_pct").reset_index(drop=True)

    print("\n" + "=" * 105)
    print("BARRIER EXPLORATION — sorted by crash_rate ascending")
    print(f"CUSUM m={config.CUSUM_THRESHOLD_MULTIPLIER}  |  "
          f"Dataset: {config.SAMPLE_ROWS:,} bars (LIGHTWEIGHT_MODE={config.LIGHTWEIGHT_MODE})")
    print("=" * 105)

    display_cols = ["setting", "pt", "sl", "horizon_bars",
                    "n_events_labeled", "crash_rate_pct",
                    "mean_horizon_min", "pct_timeout"]
    print(sorted_df[display_cols].to_string(index=False))
    print("=" * 105)

    # ── Recommendations ──────────────────────────────────────────────
    print("\nRECOMMENDATIONS:")
    targets = [
        ("~5–10% crash (very rare)",   5,  10),
        ("~15–25% crash (moderate)",  15,  25),
        ("~35–45% crash (near-balanced)", 35, 45),
        ("~50%+ crash (current-like)", 50, 100),
    ]
    for label, lo, hi in targets:
        candidates = sorted_df[
            (sorted_df["crash_rate_pct"] >= lo) &
            (sorted_df["crash_rate_pct"] <= hi)
        ]
        if len(candidates) > 0:
            row = candidates.iloc[0]
            print(f"  {label:38s} → {row['setting']:<20s} "
                  f"crash={row['crash_rate_pct']:.1f}%  "
                  f"PT={row['pt']} SL={row['sl']} H={row['horizon_bars']}")
        else:
            print(f"  {label:38s} → no setting found in this range")

    # Longest mean horizon
    longest = sorted_df.loc[sorted_df["mean_horizon_min"].idxmax()]
    print(f"  {'Longest mean_horizon (leakage study)':38s} → "
          f"{longest['setting']:<20s} "
          f"mean_horizon={longest['mean_horizon_min']:.1f} min  "
          f"PT={longest['pt']} SL={longest['sl']} H={longest['horizon_bars']}")
    print()


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    # Load df_clean
    clean_path = config.PROCESSED_DATA_DIR / "df_clean.parquet"
    logger.info("Loading df_clean from %s", clean_path)
    df_clean = pd.read_parquet(clean_path)
    logger.info("df_clean: %d bars, range %s to %s",
                len(df_clean), df_clean.index[0], df_clean.index[-1])

    # Step 1: re-run CUSUM
    events = rerun_cusum(df_clean)

    # Step 2: explore barriers
    results = explore_barriers(df_clean, events)

    # Step 3: print + save
    print_results(results)

    out_path = config.TABLES_DIR / "barrier_exploration.csv"
    results.sort_values("crash_rate_pct").to_csv(out_path, index=False)
    logger.info("Saved -> %s", out_path)
    print(f"Full table saved to: {out_path}")


if __name__ == "__main__":
    main()
