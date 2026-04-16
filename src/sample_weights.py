"""
sample_weights.py — Module 5: Concurrency-based uniqueness sample weights.

Implements the sample uniqueness weighting scheme from López de Prado (2018),
Chapter 4 — Sample Weights.

Each label horizon [event_start, t1] may overlap with other labels' horizons.
A bar t has concurrency c(t) = number of labels whose horizon contains t.
The uniqueness of a label e is the time-average of 1/c(t) over its horizon.
Labels with fewer overlapping labels receive higher uniqueness -> higher weight.

This corrects for the over-representation of information in highly overlapping
label periods, which is a form of look-ahead-free variance reduction.

Provides:
    compute_concurrency   : count of active labels per bar in close_index
    compute_sample_weights: normalized uniqueness-based weight per event
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import setup_logging

logger = setup_logging(__name__)


# ======================================================================
# CONCURRENCY
# ======================================================================

def compute_concurrency(
    event_starts: pd.DatetimeIndex,
    event_ends: pd.Series,
    close_index: pd.DatetimeIndex,
) -> pd.Series:
    """Count how many label horizons are active at each bar in close_index.

    For each bar t in close_index, concurrency c(t) is the number of events e
    whose label horizon [event_starts[e], event_ends[e]] contains t.

    Uses an O(n_events + n_bars) cumulative-sum algorithm:
        1. Mark +1 at each event start position.
        2. Mark -1 at the position immediately after each event end.
        3. Cumulative sum gives concurrency at each bar.

    Parameters
    ----------
    event_starts : pd.DatetimeIndex
        Start timestamps of label horizons (typically the event detection times).
    event_ends : pd.Series
        End timestamps of label horizons (t1 from TBM).
        Must be aligned with event_starts (same length, same order).
    close_index : pd.DatetimeIndex
        Full bar index over which concurrency is computed.
        Should span at least [event_starts.min(), event_ends.max()].

    Returns
    -------
    pd.Series
        Integer concurrency at each bar, indexed by close_index.
        Zero for bars outside all label horizons.
    """
    n_bars = len(close_index)
    changes = np.zeros(n_bars + 1, dtype=np.int32)

    starts_arr = np.asarray(event_starts, dtype="datetime64[ns]")
    ends_arr = np.asarray(event_ends, dtype="datetime64[ns]")
    idx_arr = np.asarray(close_index, dtype="datetime64[ns]")

    # searchsorted is O(log n) per call; vectorized over all events
    start_positions = np.searchsorted(idx_arr, starts_arr, side="left")
    end_positions = np.searchsorted(idx_arr, ends_arr, side="right")  # exclusive

    # Accumulate changes
    np.add.at(changes, start_positions, 1)
    np.add.at(changes, end_positions, -1)

    concurrency = pd.Series(
        np.cumsum(changes[:n_bars]),
        index=close_index,
        name="concurrency",
        dtype=np.int32,
    )

    logger.debug(
        "Concurrency: max=%d, mean=%.2f, pct_nonzero=%.2f%%",
        int(concurrency.max()),
        float(concurrency.mean()),
        100.0 * float((concurrency > 0).mean()),
    )
    return concurrency


# ======================================================================
# SAMPLE WEIGHTS
# ======================================================================

def compute_sample_weights(
    event_starts: pd.DatetimeIndex,
    event_ends: pd.Series,
    close_index: Optional[pd.DatetimeIndex] = None,
    time_decay: float = config.SAMPLE_WEIGHT_DECAY,
    min_weight: float = config.SAMPLE_WEIGHT_MIN,
) -> pd.Series:
    """Compute concurrency-based uniqueness sample weights per event.

    Algorithm (López de Prado, Ch.4):
        1. Compute concurrency c(t) at each bar using compute_concurrency.
        2. For each event e with horizon [s_e, t1_e]:
               avg_uniqueness[e] = mean(1 / c(t) for t in [s_e, t1_e] if c(t) > 0)
        3. Optionally apply exponential time decay:
               decay[e] ∝ exp(time_decay * (index_of_e / n_events - 1))
               (time_decay=0 means uniform; >0 downweights older events)
        4. weight[e] = avg_uniqueness[e] * decay[e]
        5. Normalize: weight -> weight / weight.sum() * n_events
           (so the average weight is 1.0 and weights sum to n_events)
        6. Clip from below at min_weight.

    Parameters
    ----------
    event_starts : pd.DatetimeIndex
        Start timestamps of label horizons.
    event_ends : pd.Series
        End timestamps (t1) of label horizons; aligned with event_starts.
    close_index : pd.DatetimeIndex, optional
        Full bar index.  If None, a synthetic index spanning
        [event_starts.min(), event_ends.max()] at 1-minute frequency is built.
        Providing the actual close index from df_clean is more accurate.
    time_decay : float
        Exponential decay coefficient.  0 = no decay.
        Default: config.SAMPLE_WEIGHT_DECAY.
    min_weight : float
        Floor applied after normalization.  Default: config.SAMPLE_WEIGHT_MIN.

    Returns
    -------
    pd.Series
        Sample weight per event, indexed by event_starts.
        Weights are non-negative, normalized so the mean equals 1.0.
    """
    if len(event_starts) == 0:
        logger.warning("compute_sample_weights called with empty event_starts")
        return pd.Series(dtype=float)

    # Build synthetic index if not provided
    if close_index is None:
        logger.warning(
            "close_index not provided — building synthetic 1-min index. "
            "Pass the actual df['close'].index for correct concurrency."
        )
        close_index = pd.date_range(
            start=event_starts.min(),
            end=event_ends.max(),
            freq="1min",
            tz=event_starts.tzinfo,
        )

    # --- Step 1: Concurrency ---
    concurrency = compute_concurrency(event_starts, event_ends, close_index)
    concurrency_arr = concurrency.values
    idx_arr = np.asarray(close_index, dtype="datetime64[ns]")

    starts_arr = np.asarray(event_starts, dtype="datetime64[ns]")
    ends_arr = np.asarray(event_ends, dtype="datetime64[ns]")

    start_positions = np.searchsorted(idx_arr, starts_arr, side="left")
    end_positions = np.searchsorted(idx_arr, ends_arr, side="right")  # exclusive

    # --- Step 2: Average uniqueness per event ---
    avg_uniqueness = np.empty(len(event_starts), dtype=np.float64)
    for i in range(len(event_starts)):
        s, e = start_positions[i], end_positions[i]
        c_slice = concurrency_arr[s:e].astype(np.float64)
        positive = c_slice[c_slice > 0]
        avg_uniqueness[i] = np.mean(1.0 / positive) if len(positive) > 0 else 0.0

    weights = avg_uniqueness.copy()

    # --- Step 3: Time decay ---
    if time_decay > 0.0:
        n = len(event_starts)
        time_index = np.arange(n, dtype=np.float64) / n  # 0 … 1
        decay = np.exp(time_decay * (time_index - 1.0))
        weights *= decay

    # --- Step 4–5: Normalize so mean = 1.0 (sum = n_events) ---
    total = weights.sum()
    if total > 0:
        weights = weights / total * len(event_starts)
    else:
        weights = np.ones(len(event_starts), dtype=np.float64)

    # --- Step 6: Floor ---
    if min_weight > 0.0:
        weights = np.clip(weights, min_weight, None)

    result = pd.Series(weights, index=event_starts, name="weight")
    logger.info(
        "Sample weights: n=%d, min=%.4f, mean=%.4f, max=%.4f, std=%.4f",
        len(result),
        float(result.min()),
        float(result.mean()),
        float(result.max()),
        float(result.std()),
    )
    return result
