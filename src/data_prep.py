"""
data_prep.py — Module 1: Raw data loading, cleaning, and EDA statistics.

Provides:
    load_raw_data       : parse EURUSD CSV → DatetimeIndex DataFrame
    clean_data          : remove duplicates, handle zero-volume bars
    compute_eda_stats   : summary statistics, ADF test, kurtosis/skewness
"""

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import config
from src.utils import setup_logging, timer

logger = setup_logging(__name__)


# ======================================================================
# DATA LOADING
# ======================================================================

@timer
def load_raw_data(path: Path = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw EURUSD CSV and return a clean DataFrame with DatetimeIndex.

    The CSV has NO header.  Column layout:
        0: unix_time  — POSIX timestamp (seconds)
        1: open
        2: high
        3: low
        4: close
        5: volume
        6: trade     — flag column, dropped

    Parameters
    ----------
    path : Path
        Absolute path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex (UTC), columns:
        [open, high, low, close, volume, log_return].

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    if config.LIGHTWEIGHT_MODE:
        # Count total rows (excluding header — CSV has no header so all rows are data)
        total_rows = sum(1 for _ in open(path, "r"))
        skip_rows = max(0, total_rows - config.SAMPLE_ROWS)
        logger.info(
            "LIGHTWEIGHT MODE: using last %d of %d rows (skipping first %d)",
            config.SAMPLE_ROWS,
            total_rows,
            skip_rows,
        )
        skiprows = range(1, skip_rows + 1) if skip_rows > 0 else None
    else:
        logger.info("Loading raw data from %s", path)
        skiprows = None

    df = pd.read_csv(
        path,
        header=None,
        names=["unix_time", "open", "high", "low", "close", "volume", "trade"],
        dtype={
            "unix_time": np.int64,
            "open": np.float64,
            "high": np.float64,
            "low": np.float64,
            "close": np.float64,
            "volume": np.float64,
            "trade": np.float64,
        },
        skiprows=skiprows,
    )

    # Drop trade/flag column
    df.drop(columns=["trade"], inplace=True)
    gc.collect()

    # Convert unix timestamp (seconds) to DatetimeIndex (UTC)
    df.index = pd.to_datetime(df["unix_time"], unit="s", utc=True)
    df.index.name = "datetime"
    df.drop(columns=["unix_time"], inplace=True)

    # Sort chronologically (source data may have gaps/unsorted segments)
    df.sort_index(inplace=True)

    # Compute log return: log(close_t / close_{t-1})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Convert float64 → float32 to halve memory usage
    if config.USE_FLOAT32:
        float_cols = df.select_dtypes(include="float64").columns
        df[float_cols] = df[float_cols].astype(np.float32)
        logger.info("USE_FLOAT32: converted %d columns to float32", len(float_cols))

    logger.info(
        "Loaded %d bars from %s to %s",
        len(df),
        df.index[0],
        df.index[-1],
    )

    return df


# ======================================================================
# DATA CLEANING
# ======================================================================

@timer
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove data quality issues and return cleaned DataFrame with a report.

    Steps applied in order:
    1. Remove duplicate timestamps (keep first occurrence).
    2. Remove zero-volume bars (no trading activity → unreliable OHLC).
    3. Remove rows where close price is NaN or zero.
    4. Recompute log_return after row removal to avoid erroneous gaps.
    5. Drop the first row (log_return is NaN after shift).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from :func:`load_raw_data`.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        - cleaned_df  : DataFrame with DatetimeIndex, same columns as input
        - report      : dict with keys:
            ``n_raw``, ``n_duplicate_ts``, ``n_zero_volume``,
            ``n_null_close``, ``n_clean``, ``pct_removed``
    """
    report: dict[str, Any] = {"n_raw": len(df)}
    df = df.copy()

    # Step 1: Duplicate timestamps
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_dup = n_before - len(df)
    report["n_duplicate_ts"] = n_dup
    if n_dup:
        logger.warning("Removed %d duplicate timestamp(s)", n_dup)

    # Step 2: Zero-volume bars
    n_before = len(df)
    df = df[df["volume"] > 0]
    n_zero_vol = n_before - len(df)
    report["n_zero_volume"] = n_zero_vol
    if n_zero_vol:
        logger.warning("Removed %d zero-volume bar(s)", n_zero_vol)

    # Step 3: Null or zero close prices
    n_before = len(df)
    df = df[df["close"].notna() & (df["close"] > 0)]
    n_null_close = n_before - len(df)
    report["n_null_close"] = n_null_close
    if n_null_close:
        logger.warning("Removed %d null/zero-close bar(s)", n_null_close)

    # Step 4: Recompute log_return after row removal
    df = df.sort_index()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Step 5: Drop first row (NaN log_return from shift)
    df = df.iloc[1:]

    report["n_clean"] = len(df)
    report["pct_removed"] = round(
        100.0 * (report["n_raw"] - report["n_clean"]) / report["n_raw"], 4
    )

    logger.info(
        "Cleaning complete: %d → %d rows (%.4f%% removed)",
        report["n_raw"],
        report["n_clean"],
        report["pct_removed"],
    )

    return df, report


# ======================================================================
# EDA STATISTICS
# ======================================================================

def compute_eda_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute exploratory data analysis statistics for the cleaned dataset.

    Includes descriptive stats, ADF stationarity test on log returns,
    and distributional properties relevant to flash-crash detection.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with ``log_return`` column.

    Returns
    -------
    dict[str, Any]
        Keys:
        ``n_bars``, ``date_start``, ``date_end``, ``duration_days``,
        ``log_return_mean``, ``log_return_std``, ``log_return_skew``,
        ``log_return_kurt``, ``log_return_min``, ``log_return_max``,
        ``adf_stat``, ``adf_pvalue``, ``adf_is_stationary``,
        ``close_mean``, ``close_std``, ``volume_mean``, ``volume_std``.
    """
    lr = df["log_return"].dropna()

    adf_result = adfuller(lr, autolag="AIC")

    stats: dict[str, Any] = {
        "n_bars": len(df),
        "date_start": df.index[0].isoformat(),
        "date_end": df.index[-1].isoformat(),
        "duration_days": (df.index[-1] - df.index[0]).days,
        # Log return distribution
        "log_return_mean": float(lr.mean()),
        "log_return_std": float(lr.std()),
        "log_return_skew": float(lr.skew()),
        "log_return_kurt": float(lr.kurt()),
        "log_return_min": float(lr.min()),
        "log_return_max": float(lr.max()),
        # ADF stationarity test
        "adf_stat": float(adf_result[0]),
        "adf_pvalue": float(adf_result[1]),
        "adf_is_stationary": bool(adf_result[1] < 0.05),
        # Price and volume
        "close_mean": float(df["close"].mean()),
        "close_std": float(df["close"].std()),
        "volume_mean": float(df["volume"].mean()),
        "volume_std": float(df["volume"].std()),
    }

    logger.info(
        "EDA: %d bars | return skew=%.4f kurt=%.4f | ADF p=%.4f (stationary=%s)",
        stats["n_bars"],
        stats["log_return_skew"],
        stats["log_return_kurt"],
        stats["adf_pvalue"],
        stats["adf_is_stationary"],
    )

    return stats
