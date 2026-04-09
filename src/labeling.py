"""
labeling.py — Module 3: Standard and Adaptive Triple Barrier Method (TBM).

Provides:
    apply_standard_tbm      : Label CUSUM events with fixed barrier multipliers.
    apply_adaptive_tbm      : Label events with regime-adjusted barrier multipliers.
    compare_labeling_schemes: Summarize differences between label sets (Table 4).

Label encoding:
    bin=1 (crash)    : lower barrier (stop-loss) hit first — price dropped sharply
    bin=0 (no-crash) : upper barrier (profit-taking) or vertical barrier hit first

Reference: López de Prado (2018), Advances in Financial Machine Learning,
           Chapter 3 — Labels.
"""

import gc
import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import setup_logging, timer

logger = setup_logging(__name__)


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _barrier_touch(
    close_vals: np.ndarray,
    close_idx: pd.DatetimeIndex,
    event_pos: int,
    t1_pos: int,
    upper: float,
    lower: float,
) -> tuple[pd.Timestamp, int, str]:
    """Find the first barrier touch for a single event.

    Scans bars in (event_pos, t1_pos] (exclusive start, inclusive end).
    Returns the touch timestamp, label, and which barrier was hit.

    Parameters
    ----------
    close_vals : np.ndarray
        Close prices as a numpy array (aligned with close_idx).
    close_idx : pd.DatetimeIndex
        Bar timestamps.
    event_pos : int
        Integer position of the event bar in close_idx.
    t1_pos : int
        Integer position of the vertical barrier (last bar in horizon).
        The slice is close_vals[event_pos+1 : t1_pos+1].
    upper : float
        Upper barrier level (profit-taking).
    lower : float
        Lower barrier level (stop-loss / crash detection).

    Returns
    -------
    tuple[pd.Timestamp, int, str]
        (touch_time, label, barrier_type) where:
        - label 1 = crash (lower hit), 0 = no-crash (upper or timeout)
        - barrier_type: 'sl', 'pt', or 'timeout'
    """
    path = close_vals[event_pos + 1 : t1_pos + 1]

    lower_hits = np.where(path <= lower)[0]
    upper_hits = np.where(path >= upper)[0]

    first_lower = lower_hits[0] if len(lower_hits) > 0 else None
    first_upper = upper_hits[0] if len(upper_hits) > 0 else None

    if first_lower is None and first_upper is None:
        return close_idx[t1_pos], config.NO_CRASH_LABEL, "timeout"

    if first_lower is None:
        pos = event_pos + 1 + first_upper
        return close_idx[pos], config.NO_CRASH_LABEL, "pt"

    if first_upper is None:
        pos = event_pos + 1 + first_lower
        return close_idx[pos], config.CRASH_LABEL, "sl"

    # Both barriers are present; whichever is hit first wins.
    if first_lower <= first_upper:
        pos = event_pos + 1 + first_lower
        return close_idx[pos], config.CRASH_LABEL, "sl"
    else:
        pos = event_pos + 1 + first_upper
        return close_idx[pos], config.NO_CRASH_LABEL, "pt"


# ======================================================================
# STANDARD TRIPLE BARRIER METHOD
# ======================================================================

@timer
def apply_standard_tbm(
    close: pd.Series,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    pt: float = config.TBM_PROFIT_TAKING,
    sl: float = config.TBM_STOP_LOSS,
    max_horizon: int = config.TBM_HORIZON_BARS,
) -> pd.DataFrame:
    """Apply the standard Triple Barrier Method to CUSUM-detected events.

    For each event, three barriers are constructed:
        - Upper (profit-taking) : close[event] * (1 + pt * vol[event])
        - Lower (stop-loss)     : close[event] * (1 - sl * vol[event])
        - Vertical              : event_time + max_horizon minutes

    The first barrier touched determines the label:
        - Lower hit first  → bin=1 (crash)
        - Upper hit first  → bin=0 (no-crash)
        - Vertical barrier → bin=0 (timeout / no-crash)

    Events where event_time + max_horizon exceeds the last available bar
    or where volatility is NaN/zero are dropped.

    Parameters
    ----------
    close : pd.Series
        Close price series with timezone-aware DatetimeIndex (1-min bars).
    events : pd.DatetimeIndex
        Timestamps of CUSUM-detected events.
    volatility : pd.Series
        EWM volatility series aligned with ``close``.
        Used as the barrier width: trgt = vol[event_time].
    pt : float
        Profit-taking barrier multiplier. Default: config.TBM_PROFIT_TAKING.
    sl : float
        Stop-loss barrier multiplier. Default: config.TBM_STOP_LOSS.
    max_horizon : int
        Maximum holding period in bars (minutes). Default: config.TBM_HORIZON_BARS.

    Returns
    -------
    pd.DataFrame
        Index: event timestamps (subset of ``events`` after dropping invalid ones).
        Columns:
            t1           : timestamp when the first barrier was touched
            trgt         : volatility at event time (barrier width parameter)
            bin          : label (1=crash, 0=no-crash)
            barrier_type : which barrier was hit ('sl', 'pt', 'timeout')
    """
    close_idx = close.index
    close_vals = close.values
    last_ts = close_idx[-1]
    horizon_td = pd.Timedelta(minutes=max_horizon)

    # Align volatility to the close index for fast positional lookup
    vol_aligned = volatility.reindex(close_idx)
    vol_vals = vol_aligned.values

    # Build a lookup from timestamp → integer position (O(1) per event)
    ts_to_pos: dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(close_idx)}

    rows: list[dict] = []
    n_skipped_horizon = 0
    n_skipped_vol = 0
    n_skipped_missing = 0

    for event_time in events:
        # --- Vertical barrier check ---
        t1_ts = event_time + horizon_td
        if t1_ts > last_ts:
            n_skipped_horizon += 1
            continue

        # --- Locate event bar ---
        event_pos = ts_to_pos.get(event_time)
        if event_pos is None:
            n_skipped_missing += 1
            continue

        # --- Volatility at event ---
        sigma = vol_vals[event_pos]
        if np.isnan(sigma) or sigma == 0.0:
            n_skipped_vol += 1
            continue

        c0 = close_vals[event_pos]

        # --- Barrier levels ---
        upper = c0 * (1.0 + pt * sigma)
        lower = c0 * (1.0 - sl * sigma)

        # --- Vertical barrier position (last bar <= t1_ts) ---
        t1_pos = close_idx.searchsorted(t1_ts, side="right") - 1

        # --- Find first touch ---
        touch_ts, label, barrier_type = _barrier_touch(
            close_vals, close_idx, event_pos, t1_pos, upper, lower
        )

        rows.append(
            {
                "t1": touch_ts,
                "trgt": sigma,
                "bin": label,
                "barrier_type": barrier_type,
            }
        )

    labels = pd.DataFrame(rows, index=pd.DatetimeIndex([r["t1"] for r in rows]))
    # Re-index by event time (not touch time)
    valid_events = [
        e
        for e in events
        if ts_to_pos.get(e) is not None
        and not np.isnan(vol_vals[ts_to_pos[e]])
        and vol_vals[ts_to_pos[e]] != 0.0
        and (e + horizon_td) <= last_ts
    ]
    labels.index = pd.DatetimeIndex(valid_events)
    labels.index.name = "datetime"

    del rows, valid_events
    gc.collect()

    logger.info(
        "Standard TBM: %d events → %d labeled | "
        "skipped: horizon=%d, vol=%d, missing=%d",
        len(events),
        len(labels),
        n_skipped_horizon,
        n_skipped_vol,
        n_skipped_missing,
    )
    logger.info(
        "Class distribution: crash(1)=%d (%.2f%%), no-crash(0)=%d (%.2f%%)",
        labels["bin"].sum(),
        100.0 * labels["bin"].mean(),
        (labels["bin"] == 0).sum(),
        100.0 * (1.0 - labels["bin"].mean()),
    )
    return labels


# ======================================================================
# ADAPTIVE TRIPLE BARRIER METHOD
# ======================================================================

@timer
def apply_adaptive_tbm(
    close: pd.Series,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    pt: float = config.TBM_PROFIT_TAKING,
    sl: float = config.TBM_STOP_LOSS,
    max_horizon: int = config.TBM_HORIZON_BARS,
    regime_window: int = config.TBM_ADAPTIVE_VOL_LOOKBACK,
    high_vol_pct: float = 75.0,
    low_vol_pct: float = 25.0,
    high_vol_factor: float = config.TBM_ADAPTIVE_PT_HIGH_VOL / config.TBM_PROFIT_TAKING,
    low_vol_factor: float = config.TBM_ADAPTIVE_PT_LOW_VOL / config.TBM_PROFIT_TAKING,
) -> pd.DataFrame:
    """Apply the Adaptive Triple Barrier Method with regime-adjusted barriers.

    Step 1 — Regime detection:
        Computes a rolling percentile rank of volatility over ``regime_window``
        bars (using only past data, no lookahead).  At each event:
            rank > high_vol_pct/100 → HIGH   regime → multiply pt & sl by high_vol_factor
            rank < low_vol_pct/100  → LOW    regime → multiply pt & sl by low_vol_factor
            otherwise               → NORMAL regime → unchanged multipliers

    Step 2 — Barrier construction (same as standard TBM but with pt_adj / sl_adj).

    The regime logic reflects that high-volatility environments require
    relatively smaller multipliers to remain sensitive to crash events, while
    low-volatility environments require wider barriers to avoid spurious labels.

    Parameters
    ----------
    close : pd.Series
        Close price series with timezone-aware DatetimeIndex (1-min bars).
    events : pd.DatetimeIndex
        Timestamps of CUSUM-detected events.
    volatility : pd.Series
        EWM volatility series aligned with ``close``.
    pt : float
        Base profit-taking multiplier. Adjusted per regime.
    sl : float
        Base stop-loss multiplier. Adjusted per regime.
    max_horizon : int
        Maximum holding period in bars (minutes).
    regime_window : int
        Rolling window (bars) for volatility percentile rank computation.
    high_vol_pct : float
        Percentile threshold (0–100) above which the regime is HIGH.
    low_vol_pct : float
        Percentile threshold (0–100) below which the regime is LOW.
    high_vol_factor : float
        Multiplicative factor applied to pt and sl in HIGH-vol regime.
        Default derived from config.TBM_ADAPTIVE_PT_HIGH_VOL / TBM_PROFIT_TAKING.
    low_vol_factor : float
        Multiplicative factor applied to pt and sl in LOW-vol regime.
        Default derived from config.TBM_ADAPTIVE_PT_LOW_VOL / TBM_PROFIT_TAKING.

    Returns
    -------
    pd.DataFrame
        Same schema as ``apply_standard_tbm`` plus:
            regime   : volatility regime ('high', 'low', 'normal')
            pt_adj   : effective profit-taking multiplier used
            sl_adj   : effective stop-loss multiplier used
    """
    close_idx = close.index
    close_vals = close.values
    last_ts = close_idx[-1]
    horizon_td = pd.Timedelta(minutes=max_horizon)

    # Align volatility to the close index
    vol_aligned = volatility.reindex(close_idx)
    vol_vals = vol_aligned.values

    # --- Step 1: Rolling percentile rank of volatility (no lookahead) ---
    logger.info(
        "Computing rolling percentile rank (window=%d) for regime detection ...",
        regime_window,
    )
    rolling_rank = vol_aligned.rolling(regime_window, min_periods=regime_window).rank(pct=True)
    rank_vals = rolling_rank.values

    # Threshold fractions
    high_threshold = high_vol_pct / 100.0
    low_threshold = low_vol_pct / 100.0

    ts_to_pos: dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(close_idx)}

    rows: list[dict] = []
    n_skipped_horizon = 0
    n_skipped_vol = 0
    n_skipped_missing = 0
    n_skipped_rank = 0

    for event_time in events:
        t1_ts = event_time + horizon_td
        if t1_ts > last_ts:
            n_skipped_horizon += 1
            continue

        event_pos = ts_to_pos.get(event_time)
        if event_pos is None:
            n_skipped_missing += 1
            continue

        sigma = vol_vals[event_pos]
        if np.isnan(sigma) or sigma == 0.0:
            n_skipped_vol += 1
            continue

        rank = rank_vals[event_pos]
        if np.isnan(rank):
            n_skipped_rank += 1
            continue

        # --- Determine regime and adjusted multipliers ---
        if rank > high_threshold:
            regime = "high"
            pt_adj = pt * high_vol_factor
            sl_adj = sl * high_vol_factor
        elif rank < low_threshold:
            regime = "low"
            pt_adj = pt * low_vol_factor
            sl_adj = sl * low_vol_factor
        else:
            regime = "normal"
            pt_adj = pt
            sl_adj = sl

        c0 = close_vals[event_pos]
        upper = c0 * (1.0 + pt_adj * sigma)
        lower = c0 * (1.0 - sl_adj * sigma)

        t1_pos = close_idx.searchsorted(t1_ts, side="right") - 1

        touch_ts, label, barrier_type = _barrier_touch(
            close_vals, close_idx, event_pos, t1_pos, upper, lower
        )

        rows.append(
            {
                "t1": touch_ts,
                "trgt": sigma,
                "bin": label,
                "barrier_type": barrier_type,
                "regime": regime,
                "pt_adj": pt_adj,
                "sl_adj": sl_adj,
            }
        )

    # Build event-indexed DataFrame
    valid_events = [
        e
        for e in events
        if ts_to_pos.get(e) is not None
        and not np.isnan(vol_vals[ts_to_pos[e]])
        and vol_vals[ts_to_pos[e]] != 0.0
        and not np.isnan(rank_vals[ts_to_pos[e]])
        and (e + horizon_td) <= last_ts
    ]
    labels = pd.DataFrame(rows)
    labels.index = pd.DatetimeIndex(valid_events)
    labels.index.name = "datetime"

    del rows, valid_events, rolling_rank, rank_vals
    gc.collect()

    # Regime breakdown
    regime_counts = labels["regime"].value_counts()
    logger.info(
        "Adaptive TBM: %d events → %d labeled | "
        "skipped: horizon=%d, vol=%d, rank=%d, missing=%d",
        len(events),
        len(labels),
        n_skipped_horizon,
        n_skipped_vol,
        n_skipped_rank,
        n_skipped_missing,
    )
    logger.info("Regime breakdown: %s", regime_counts.to_dict())
    logger.info(
        "Class distribution: crash(1)=%d (%.2f%%), no-crash(0)=%d (%.2f%%)",
        labels["bin"].sum(),
        100.0 * labels["bin"].mean(),
        (labels["bin"] == 0).sum(),
        100.0 * (1.0 - labels["bin"].mean()),
    )
    return labels


# ======================================================================
# COMPARISON
# ======================================================================

def compare_labeling_schemes(
    labels_standard: pd.DataFrame,
    labels_adaptive: pd.DataFrame,
) -> pd.DataFrame:
    """Compare standard and adaptive TBM label sets.

    Computes event counts, class balance, label agreement rate, and
    barrier type distribution for each scheme.  Intended for Table 4.

    Parameters
    ----------
    labels_standard : pd.DataFrame
        Output of ``apply_standard_tbm``.
    labels_adaptive : pd.DataFrame
        Output of ``apply_adaptive_tbm``.

    Returns
    -------
    pd.DataFrame
        Summary comparison DataFrame with rows: standard, adaptive.
        Columns: n_events, n_crash, n_no_crash, crash_rate_pct,
                 pct_sl, pct_pt, pct_timeout, [agreement_pct, regime breakdown].
    """
    def _summarize(lbl: pd.DataFrame, name: str) -> dict:
        n = len(lbl)
        n_crash = int(lbl["bin"].sum())
        n_no_crash = n - n_crash
        bt = lbl["barrier_type"].value_counts(normalize=True) * 100
        row = {
            "scheme": name,
            "n_events": n,
            "n_crash": n_crash,
            "n_no_crash": n_no_crash,
            "crash_rate_pct": round(100.0 * n_crash / n, 4) if n > 0 else 0.0,
            "pct_sl": round(bt.get("sl", 0.0), 2),
            "pct_pt": round(bt.get("pt", 0.0), 2),
            "pct_timeout": round(bt.get("timeout", 0.0), 2),
        }
        if "regime" in lbl.columns:
            rc = lbl["regime"].value_counts(normalize=True) * 100
            row["pct_regime_high"] = round(rc.get("high", 0.0), 2)
            row["pct_regime_low"] = round(rc.get("low", 0.0), 2)
            row["pct_regime_normal"] = round(rc.get("normal", 0.0), 2)
        return row

    rows = [
        _summarize(labels_standard, "standard"),
        _summarize(labels_adaptive, "adaptive"),
    ]

    # Label agreement on shared events
    shared_idx = labels_standard.index.intersection(labels_adaptive.index)
    if len(shared_idx) > 0:
        agreement = (
            labels_standard.loc[shared_idx, "bin"]
            == labels_adaptive.loc[shared_idx, "bin"]
        ).mean() * 100.0
        for row in rows:
            row["n_shared_events"] = len(shared_idx)
            row["label_agreement_pct"] = round(agreement, 2)
    else:
        for row in rows:
            row["n_shared_events"] = 0
            row["label_agreement_pct"] = float("nan")

    comparison = pd.DataFrame(rows).set_index("scheme")
    logger.info("Labeling comparison:\n%s", comparison.to_string())
    return comparison


# ======================================================================
# MAIN — ORCHESTRATION
# ======================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(config.PROJECT_ROOT))
    from src.utils import load_result, save_result
    from src.cusum import compute_ewm_volatility

    logger.info("=" * 60)
    logger.info("Phase 2: Triple Barrier Labeling")
    logger.info("=" * 60)

    # --- Load data ---
    logger.info("Loading df_clean and cusum_events ...")
    df = load_result("df_clean")
    events_df = load_result("cusum_events")
    events: pd.DatetimeIndex = events_df.index

    close: pd.Series = df["close"]
    log_returns: pd.Series = df["log_return"]

    # --- Compute EWM volatility (same span used for CUSUM) ---
    logger.info("Computing EWM volatility (span=%d) ...", config.TBM_VOLATILITY_SPAN)
    volatility = compute_ewm_volatility(log_returns, span=config.TBM_VOLATILITY_SPAN)

    # ── Standard TBM ──────────────────────────────────────────────────
    logger.info(
        "Running standard TBM: PT=%.1f, SL=%.1f, horizon=%d bars ...",
        config.TBM_PROFIT_TAKING,
        config.TBM_STOP_LOSS,
        config.TBM_HORIZON_BARS,
    )
    labels_standard = apply_standard_tbm(
        close=close,
        events=events,
        volatility=volatility,
        pt=config.TBM_PROFIT_TAKING,
        sl=config.TBM_STOP_LOSS,
        max_horizon=config.TBM_HORIZON_BARS,
    )

    print("\n=== Standard TBM Class Distribution ===")
    print(labels_standard["bin"].value_counts().rename({1: "crash", 0: "no-crash"}))
    print(labels_standard["barrier_type"].value_counts())

    # ── Adaptive TBM ──────────────────────────────────────────────────
    high_factor = config.TBM_ADAPTIVE_PT_HIGH_VOL / config.TBM_PROFIT_TAKING
    low_factor = config.TBM_ADAPTIVE_PT_LOW_VOL / config.TBM_PROFIT_TAKING

    logger.info(
        "Running adaptive TBM: regime_window=%d, high_factor=%.3f, low_factor=%.3f ...",
        config.TBM_ADAPTIVE_VOL_LOOKBACK,
        high_factor,
        low_factor,
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

    print("\n=== Adaptive TBM Class Distribution ===")
    print(labels_adaptive["bin"].value_counts().rename({1: "crash", 0: "no-crash"}))
    print(labels_adaptive["barrier_type"].value_counts())
    if "regime" in labels_adaptive.columns:
        print(labels_adaptive["regime"].value_counts())

    # ── Comparison ───────────────────────────────────────────────────
    comparison = compare_labeling_schemes(labels_standard, labels_adaptive)
    print("\n=== Labeling Scheme Comparison ===")
    print(comparison.T.to_string())

    # ── Sample weights (requires sample_weights module) ───────────────
    logger.info("Computing sample weights ...")
    from src.sample_weights import compute_sample_weights

    weights_standard = compute_sample_weights(
        event_starts=labels_standard.index,
        event_ends=labels_standard["t1"],
        close_index=close.index,
    )
    labels_standard["weight"] = weights_standard

    weights_adaptive = compute_sample_weights(
        event_starts=labels_adaptive.index,
        event_ends=labels_adaptive["t1"],
        close_index=close.index,
    )
    labels_adaptive["weight"] = weights_adaptive

    print("\n=== Sample Weights (standard) ===")
    print(weights_standard.describe())
    print("\n=== Sample Weights (adaptive) ===")
    print(weights_adaptive.describe())

    # ── Save ─────────────────────────────────────────────────────────
    save_result(labels_standard, "labels_standard")
    save_result(labels_adaptive, "labels_adaptive")
    save_result(comparison.reset_index(), "labeling_comparison")

    logger.info("Phase 2 complete.  Outputs saved to data/processed/.")
