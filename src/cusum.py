"""
cusum.py — Module 2: CUSUM-based event filter for irregularly spaced events.

Provides:
    compute_ewm_volatility    : exponentially weighted moving std of log returns
    cusum_filter              : symmetric CUSUM filter -> DatetimeIndex of events
    run_sensitivity_analysis  : event rate across threshold multipliers

Reference: López de Prado (2018), Advances in Financial Machine Learning,
           Chapter 2 — The CUSUM Filter.
"""

import gc
import logging
from typing import Union

import numpy as np
import pandas as pd

import config
from src.utils import setup_logging, timer

logger = setup_logging(__name__)


# ======================================================================
# EWM VOLATILITY
# ======================================================================

def compute_ewm_volatility(
    log_returns: pd.Series,
    span: int = config.CUSUM_LOOKBACK,
) -> pd.Series:
    """Compute exponentially weighted moving standard deviation of log returns.

    Used as the dynamic threshold baseline for the CUSUM filter.
    Volatility at time t is estimated using only data up to t (no lookahead).

    Parameters
    ----------
    log_returns : pd.Series
        Series of log returns with DatetimeIndex.
    span : int
        EWM span (half-life equivalent).  Defaults to config.CUSUM_LOOKBACK.

    Returns
    -------
    pd.Series
        EWM standard deviation, same index as ``log_returns``.
        First ``span`` values will be NaN (insufficient history).
    """
    ewm_std = log_returns.ewm(span=span, min_periods=span).std()
    logger.debug(
        "EWM volatility computed: span=%d, non-null=%d / %d",
        span,
        ewm_std.notna().sum(),
        len(ewm_std),
    )
    return ewm_std


# ======================================================================
# CUSUM FILTER
# ======================================================================

@timer
def cusum_filter(
    log_returns: pd.Series,
    threshold: Union[pd.Series, float],
) -> pd.DatetimeIndex:
    """Apply the symmetric CUSUM filter to detect structural breaks.

    The filter accumulates positive and negative deviations separately.
    An event is triggered when either accumulator crosses the threshold,
    after which both accumulators reset to zero.

    Algorithm (corrected from legacy code):
        s_pos = max(0, s_pos + diff)
        s_neg = min(0, s_neg + diff)
        if s_pos >= h or s_neg <= -h:
            record event, reset s_pos = s_neg = 0

    Parameters
    ----------
    log_returns : pd.Series
        Series of log returns with DatetimeIndex.
    threshold : pd.Series or float
        Detection threshold h.  If a Series, must be aligned with
        ``log_returns`` (dynamic threshold, e.g. 2.0 * EWM volatility).
        If a float, applies a fixed threshold across all bars.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps of detected events (irregularly spaced).
    """
    # Align threshold to log_returns index
    if isinstance(threshold, (int, float)):
        h = pd.Series(threshold, index=log_returns.index)
    else:
        h = threshold.reindex(log_returns.index)

    events: list[pd.Timestamp] = []
    s_pos: float = 0.0
    s_neg: float = 0.0

    # Drop leading NaNs (common when threshold is EWM-based)
    valid_mask = log_returns.notna() & h.notna()
    if not valid_mask.any():
        logger.warning("cusum_filter: no valid (non-NaN) rows — returning empty index")
        return pd.DatetimeIndex([])

    for ts, diff in log_returns[valid_mask].items():
        h_t = h.loc[ts]

        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)

        if s_pos >= h_t or s_neg <= -h_t:
            events.append(ts)
            s_pos = 0.0
            s_neg = 0.0

    event_idx = pd.DatetimeIndex(events)
    logger.info(
        "CUSUM filter: %d events detected from %d valid bars (%.2f%%)",
        len(event_idx),
        valid_mask.sum(),
        100.0 * len(event_idx) / valid_mask.sum() if valid_mask.sum() else 0.0,
    )
    # Free temporary accumulator list
    del events
    gc.collect()
    return event_idx


# ======================================================================
# SENSITIVITY ANALYSIS
# ======================================================================

@timer
def run_sensitivity_analysis(
    log_returns: pd.Series,
    volatility: pd.Series,
    multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """Run CUSUM filter across a range of threshold multipliers.

    For each multiplier m, threshold = m * volatility.
    Reports the event count and event rate (% of valid bars).

    Parameters
    ----------
    log_returns : pd.Series
        Series of log returns with DatetimeIndex.
    volatility : pd.Series
        EWM volatility series aligned with ``log_returns``.
    multipliers : list[float], optional
        Threshold multipliers to test.
        Defaults to [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0].

    Returns
    -------
    pd.DataFrame
        Columns: ``multiplier``, ``n_events``, ``n_valid_bars``,
        ``event_rate_pct``.
        Sorted by multiplier ascending.
    """
    if multipliers is None:
        multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    valid_bars = int((log_returns.notna() & volatility.notna()).sum())
    rows = []

    for m in multipliers:
        threshold = m * volatility
        events = cusum_filter(log_returns, threshold)
        event_rate = 100.0 * len(events) / valid_bars if valid_bars else 0.0
        rows.append(
            {
                "multiplier": m,
                "n_events": len(events),
                "n_valid_bars": valid_bars,
                "event_rate_pct": round(event_rate, 4),
            }
        )
        logger.info(
            "Sensitivity m=%.1f -> %d events (%.2f%%)", m, len(events), event_rate
        )

    result = pd.DataFrame(rows).sort_values("multiplier").reset_index(drop=True)
    return result
