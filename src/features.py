"""
features.py — Module 4: Feature Engineering (18 features, 4 groups).

Provides:
    engineer_features   : Compute all 18 features on the full price DataFrame.
    get_feature_matrix  : Extract feature values at event timestamps, drop NaN rows.

Feature groups (all justified by literature review — see CLAUDE.md):
    Group 1 — Volatility & Range    (6 features)
    Group 2 — Microstructure Proxies (4 features)
    Group 3 — Momentum & Return     (4 features)
    Group 4 — Trend & Seasonality   (4 features)

Memory constraints (MacBook Air M1 8GB):
    - All feature columns stored as float32.
    - Computation is sequential; intermediate arrays are released promptly.

Reference: López de Prado (2018), AFML, Chapters 4 & 17;
           Christensen et al. (2025); Ranaldo & Somogyi (2021);
           Corwin & Schultz (2012).
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
# GROUP 1 — VOLATILITY & RANGE
# ======================================================================

def _compute_volatility(log_return: pd.Series) -> pd.Series:
    """EWM standard deviation of log-returns (span=50).

    Reference: Bollerslev et al. 2018; Ardia et al. 2018.
    """
    return (
        log_return
        .ewm(span=config.VOLATILITY_EWM_SPAN, min_periods=config.VOLATILITY_EWM_SPAN)
        .std()
        .astype("float32")
    )


def _compute_garman_klass_vol(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
) -> pd.Series:
    """Garman-Klass range-based volatility estimator, rolling mean over 20 bars.

    Formula: GK = 0.5 * log(H/L)^2 - (2*ln(2) - 1) * log(C/O)^2
    Then take rolling mean over GK_VOL_WINDOW bars.

    Reference: Fiszeder et al. 2019.
    """
    hl = np.log(high / low).astype("float64") ** 2
    co = np.log(close / open_).astype("float64") ** 2
    gk = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    return (
        gk
        .rolling(window=config.GK_VOL_WINDOW, min_periods=config.GK_VOL_WINDOW)
        .mean()
        .astype("float32")
    )


def _compute_high_low_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """(High - Low) / Close, rolling mean over 20 bars.

    Reference: Fiszeder et al. 2019.
    """
    raw = ((high - low) / close).astype("float64")
    return (
        raw
        .rolling(window=config.HIGH_LOW_WINDOW, min_periods=config.HIGH_LOW_WINDOW)
        .mean()
        .astype("float32")
    )


def _compute_bb_width(close: pd.Series) -> pd.Series:
    """Bollinger Band width = (upper - lower) / MA_20.

    upper = MA_20 + BB_STD_MULTIPLIER * std_20
    lower = MA_20 - BB_STD_MULTIPLIER * std_20

    Reference: Christensen et al. 2025.
    """
    w = config.BB_WINDOW
    ma = close.rolling(window=w, min_periods=w).mean()
    std = close.rolling(window=w, min_periods=w).std()
    upper = ma + config.BB_STD_MULTIPLIER * std
    lower = ma - config.BB_STD_MULTIPLIER * std
    bb_width = ((upper - lower) / ma).astype("float32")
    return bb_width


def _compute_roll_skew(log_return: pd.Series) -> pd.Series:
    """Rolling skewness of log-return over 60 bars.

    Reference: Christensen et al. 2025.
    """
    return (
        log_return
        .rolling(window=config.ROLL_SKEW_WINDOW, min_periods=config.ROLL_SKEW_WINDOW)
        .skew()
        .astype("float32")
    )


def _compute_roll_kurt(log_return: pd.Series) -> pd.Series:
    """Rolling excess kurtosis of log-return over 60 bars.

    Reference: Christensen et al. 2025.
    """
    return (
        log_return
        .rolling(window=config.ROLL_KURT_WINDOW, min_periods=config.ROLL_KURT_WINDOW)
        .kurt()
        .astype("float32")
    )


# ======================================================================
# GROUP 2 — MICROSTRUCTURE PROXIES
# ======================================================================

def _compute_amihud(log_return: pd.Series, volume: pd.Series, close: pd.Series) -> pd.Series:
    """Amihud illiquidity ratio: |log_return| / (volume * close + epsilon).

    Reference: Ranaldo & Somogyi 2021.
    """
    amihud = (
        log_return.abs().astype("float64")
        / (volume.astype("float64") * close.astype("float64") + config.AMIHUD_EPSILON)
    )
    return amihud.astype("float32")


def _compute_cs_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin-Schultz bid-ask spread estimator.

    Steps per Corwin & Schultz (2012), eq. (14):
        beta_t  = [log(H_t/L_t)]^2 + [log(H_{t+1}/L_{t+1})]^2
        gamma_t = [log(max(H_t, H_{t+1}) / min(L_t, L_{t+1}))]^2
        alpha_t = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
                  - sqrt(gamma / (3 - 2*sqrt(2)))
        spread_t = 2*(exp(alpha_t) - 1) / (1 + exp(alpha_t))   clipped at 0

    Reference: Cenedese et al. 2021.
    """
    log_hl = np.log(high / low).astype("float64")
    beta = log_hl ** 2 + log_hl.shift(1) ** 2

    h2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h2 / l2).astype("float64") ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)          # ≈ 0.1716
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = spread.clip(lower=0.0)        # negative estimates are noise -> 0
    return spread.astype("float32")


def _compute_volume_change_ratio(volume: pd.Series) -> pd.Series:
    """volume / rolling_mean_volume(20) - 1.

    Reference: Piccotti & Schreiber 2020.
    """
    roll_vol = volume.rolling(
        window=config.VOLUME_MA_WINDOW, min_periods=config.VOLUME_MA_WINDOW
    ).mean()
    vcr = (volume.astype("float64") / roll_vol.astype("float64")) - 1.0
    return vcr.astype("float32")


def _compute_interact_vol_amihud(
    volatility: pd.Series, amihud: pd.Series
) -> pd.Series:
    """Interaction term: volatility × amihud.

    Captures simultaneous high-volatility and low-liquidity stress states.
    Reference: Breedon et al. 2023.
    """
    return (volatility.astype("float64") * amihud.astype("float64")).astype("float32")


# ======================================================================
# GROUP 3 — MOMENTUM & RETURN
# ======================================================================

def _compute_log_return(close: pd.Series) -> pd.Series:
    """log(close_t / close_{t-1}).

    Reference: Gu et al. 2020.
    """
    return np.log(close / close.shift(1)).astype("float32")


def _compute_return_lag_1(log_return: pd.Series) -> pd.Series:
    """Lagged log-return: log_return.shift(1).

    Reference: Christensen et al. 2025.
    """
    return log_return.shift(1).astype("float32")


def _compute_rsi(log_return: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index over `period` bars.

    Uses Wilder's smoothing (equivalent to EWM alpha=1/period, adjust=False).
    """
    delta = log_return.astype("float64")
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.astype("float32")


def _compute_efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Fractal Efficiency Ratio: |net_change(w)| / sum(|bar_changes(w)|).

    ER -> 1 means directional (trending); ER -> 0 means choppy (noisy).
    """
    net_change = (close - close.shift(window)).abs().astype("float64")
    bar_changes = close.diff().abs().astype("float64")
    total_path = bar_changes.rolling(window=window, min_periods=window).sum()
    er = net_change / (total_path + 1e-9)
    return er.astype("float32")


# ======================================================================
# GROUP 4 — TREND & SEASONALITY
# ======================================================================

def _compute_ema_deviation(close: pd.Series, window: int) -> pd.Series:
    """(close - EMA_w) / EMA_w."""
    ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
    dev = ((close.astype("float64") - ema.astype("float64")) / ema.astype("float64"))
    return dev.astype("float32")


def _compute_vwap_deviation(
    close: pd.Series, volume: pd.Series, window: int
) -> pd.Series:
    """(close - VWAP_w) / VWAP_w  where VWAP uses a rolling window.

    VWAP_w = sum(close * volume, w) / sum(volume, w).
    """
    cv = (close.astype("float64") * volume.astype("float64"))
    vwap = (
        cv.rolling(window=window, min_periods=window).sum()
        / volume.astype("float64").rolling(window=window, min_periods=window).sum()
    )
    dev = (close.astype("float64") - vwap) / (vwap + 1e-9)
    return dev.astype("float32")


def _compute_time_features(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Cyclic encoding of intraday time (minutes since midnight).

    sin_t = sin(2π * minutes_in_day / 1440)
    cos_t = cos(2π * minutes_in_day / 1440)

    Reference: Hasbrouck & Levich 2021.
    """
    minutes = (index.hour * 60 + index.minute).astype("float64")
    angle = 2.0 * np.pi * minutes / config.MINUTES_IN_DAY
    time_sin = pd.Series(np.sin(angle).astype("float32"), index=index, name="time_sin")
    time_cos = pd.Series(np.cos(angle).astype("float32"), index=index, name="time_cos")
    return time_sin, time_cos


# ======================================================================
# PUBLIC API
# ======================================================================

@timer
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 18 features and append them to a copy of df.

    Expects df to have columns: open, high, low, close, volume, log_return
    (log_return is added by data_prep.py).

    All feature columns are stored as float32 to minimise memory usage.
    No look-ahead: every rolling/EWM window is strictly backward-looking.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame with DatetimeIndex and a ``log_return`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 18 new feature columns appended.
        The original OHLCV columns are preserved.
    """
    logger.info(
        "engineer_features: input shape=%s, mode=%s",
        df.shape,
        "LIGHTWEIGHT" if config.LIGHTWEIGHT_MODE else "FULL",
    )

    out = df.copy()
    close  = out["close"]
    high   = out["high"]
    low    = out["low"]
    open_  = out["open"]
    volume = out["volume"]
    lr     = out["log_return"]

    # ------------------------------------------------------------------
    # Group 1 — Volatility & Range
    # ------------------------------------------------------------------
    logger.info("Computing Group 1 — Volatility & Range …")
    out["volatility"]       = _compute_volatility(lr)
    out["garman_klass_vol"] = _compute_garman_klass_vol(high, low, close, open_)
    out["high_low_range"]   = _compute_high_low_range(high, low, close)
    out["bb_width"]         = _compute_bb_width(close)
    out["roll_skew"]        = _compute_roll_skew(lr)
    out["roll_kurt"]        = _compute_roll_kurt(lr)

    # ------------------------------------------------------------------
    # Group 2 — Microstructure Proxies
    # ------------------------------------------------------------------
    logger.info("Computing Group 2 — Microstructure Proxies …")
    out["amihud"]              = _compute_amihud(lr, volume, close)
    out["cs_spread"]           = _compute_cs_spread(high, low)
    out["volume_change_ratio"] = _compute_volume_change_ratio(volume)
    out["interact_vol_amihud"] = _compute_interact_vol_amihud(
        out["volatility"], out["amihud"]
    )

    # ------------------------------------------------------------------
    # Group 3 — Momentum & Return
    # ------------------------------------------------------------------
    logger.info("Computing Group 3 — Momentum & Return …")
    # log_return already exists; cast to float32 for consistency
    out["log_return"]       = out["log_return"].astype("float32")
    out["return_lag_1"]     = _compute_return_lag_1(lr)
    out["rsi_14"]           = _compute_rsi(lr, period=config.RSI_PERIOD)
    out["efficiency_ratio"] = _compute_efficiency_ratio(
        close, window=config.EFFICIENCY_RATIO_WINDOW
    )

    # ------------------------------------------------------------------
    # Group 4 — Trend & Seasonality
    # ------------------------------------------------------------------
    logger.info("Computing Group 4 — Trend & Seasonality …")
    out["ema_deviation"]  = _compute_ema_deviation(close, window=config.EMA_WINDOW)
    out["vwap_deviation"] = _compute_vwap_deviation(
        close, volume, window=config.VWAP_WINDOW
    )
    out["time_sin"], out["time_cos"] = _compute_time_features(out.index)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    n_features_computed = sum(c in out.columns for c in config.FEATURE_NAMES)
    logger.info(
        "engineer_features: %d / %d features computed  (output shape=%s)",
        n_features_computed,
        config.N_FEATURES,
        out.shape,
    )

    if n_features_computed != config.N_FEATURES:
        missing = [c for c in config.FEATURE_NAMES if c not in out.columns]
        logger.error("Missing features: %s", missing)
        raise RuntimeError(f"Expected {config.N_FEATURES} features; missing: {missing}")

    return out


@timer
def get_feature_matrix(
    df_features: pd.DataFrame,
    events: pd.DataFrame,
    feature_list: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract feature values at event timestamps.

    Aligns df_features with the event index (inner join), then drops rows
    that contain NaN or ±inf in any feature column.  Logs a warning with
    the count of dropped rows.

    Parameters
    ----------
    df_features : pd.DataFrame
        Full-bar feature DataFrame produced by ``engineer_features``.
    events : pd.DataFrame
        Event timestamps as index (from cusum / labeling modules).
    feature_list : list[str], optional
        Subset of features to extract.  Defaults to config.FEATURE_NAMES.

    Returns
    -------
    pd.DataFrame
        Feature matrix with shape (n_events_retained, n_features).
        Index = event timestamps.
    """
    if feature_list is None:
        feature_list = config.FEATURE_NAMES

    missing_cols = [c for c in feature_list if c not in df_features.columns]
    if missing_cols:
        raise ValueError(f"Requested features not in df_features: {missing_cols}")

    # Align to event timestamps (inner join)
    X = df_features.loc[df_features.index.isin(events.index), feature_list].copy()
    logger.info(
        "get_feature_matrix: events=%d, aligned=%d",
        len(events),
        len(X),
    )

    if len(X) < len(events):
        unmatched = len(events) - len(X)
        logger.warning(
            "%d event timestamps not found in df_features (possible index mismatch).",
            unmatched,
        )

    # Replace ±inf with NaN then drop
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    n_before = len(X)
    X.dropna(inplace=True)
    n_dropped = n_before - len(X)

    if n_dropped > 0:
        logger.warning(
            "Dropped %d rows (%.2f%%) containing NaN/inf from feature matrix.",
            n_dropped,
            100.0 * n_dropped / n_before if n_before > 0 else 0.0,
        )
    else:
        logger.info("No NaN/inf rows dropped — feature matrix is clean.")

    logger.info("get_feature_matrix: final shape=%s", X.shape)
    return X


# ======================================================================
# ORCHESTRATION (run as script)
# ======================================================================

def _print_feature_group_summary(X: pd.DataFrame) -> None:
    """Print per-group feature statistics."""
    groups = {
        "Group 1 — Volatility & Range":     ["volatility", "garman_klass_vol", "high_low_range", "bb_width", "roll_skew", "roll_kurt"],
        "Group 2 — Microstructure Proxies": ["amihud", "cs_spread", "volume_change_ratio", "interact_vol_amihud"],
        "Group 3 — Momentum & Return":      ["log_return", "return_lag_1", "rsi_14", "efficiency_ratio"],
        "Group 4 — Trend & Seasonality":    ["ema_deviation", "vwap_deviation", "time_sin", "time_cos"],
    }
    print("\n" + "=" * 72)
    print("FEATURE GROUP SUMMARY")
    print("=" * 72)
    for group_name, feats in groups.items():
        present = [f for f in feats if f in X.columns]
        print(f"\n{group_name}  ({len(present)} features)")
        print(X[present].describe().round(6).to_string())
    print("=" * 72)


def _check_nan_inf(X: pd.DataFrame, tag: str = "") -> None:
    """Log and print NaN / inf counts per feature."""
    inf_mask  = X.isin([np.inf, -np.inf])
    nan_mask  = X.isna()
    n_inf     = inf_mask.sum().sum()
    n_nan     = nan_mask.sum().sum()
    print(f"\n[{tag}] NaN count: {n_nan}  |  Inf count: {n_inf}")
    if n_nan > 0:
        print("  NaN per column:\n", nan_mask.sum()[nan_mask.sum() > 0].to_string())
    if n_inf > 0:
        print("  Inf per column:\n", inf_mask.sum()[inf_mask.sum() > 0].to_string())


if __name__ == "__main__":
    import sys
    from src.utils import load_result, save_result

    # ------------------------------------------------------------------
    # 1. Load df_clean and labels
    # ------------------------------------------------------------------
    print("Loading df_clean.parquet …")
    df_clean = load_result("df_clean")
    print(f"  df_clean shape: {df_clean.shape}")

    print("Loading labels_adaptive.parquet …")
    labels_adaptive = load_result("labels_adaptive")
    print(f"  labels_adaptive shape: {labels_adaptive.shape}")

    print("Loading labels_standard.parquet …")
    labels_standard = load_result("labels_standard")
    print(f"  labels_standard shape: {labels_standard.shape}")

    # ------------------------------------------------------------------
    # 2. Engineer features on full df_clean
    # ------------------------------------------------------------------
    print("\nRunning engineer_features …")
    df_feat = engineer_features(df_clean)
    del df_clean
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Extract feature matrix aligned to event timestamps
    # ------------------------------------------------------------------
    print("\nExtracting feature matrix (adaptive events) …")
    X_adaptive = get_feature_matrix(df_feat, labels_adaptive)

    print("\nExtracting feature matrix (standard events) …")
    X_standard = get_feature_matrix(df_feat, labels_standard)

    # ------------------------------------------------------------------
    # 4. Merge features + labels (+weights if available)
    # ------------------------------------------------------------------
    def _build_dataset(
        X: pd.DataFrame,
        labels: pd.DataFrame,
        tag: str,
    ) -> pd.DataFrame:
        common_idx = X.index.intersection(labels.index)
        logger.info(
            "%s: X=%d, labels=%d, intersection=%d",
            tag, len(X), len(labels), len(common_idx),
        )
        X_aln = X.loc[common_idx]
        y_aln = labels.loc[common_idx]
        merged = X_aln.copy()
        # Add label columns that exist in labels DataFrame
        for col in ["bin", "t1", "ret"]:
            if col in y_aln.columns:
                merged[col] = y_aln[col].values
        # Add sample weights if present
        if "weight" in y_aln.columns:
            merged["weight"] = y_aln["weight"].values
        return merged

    print("\nBuilding X_y_adaptive …")
    xy_adaptive = _build_dataset(X_adaptive, labels_adaptive, "adaptive")

    print("\nBuilding X_y_standard …")
    xy_standard = _build_dataset(X_standard, labels_standard, "standard")

    # ------------------------------------------------------------------
    # 5. Quality checks
    # ------------------------------------------------------------------
    _print_feature_group_summary(X_adaptive)
    _check_nan_inf(X_adaptive, "adaptive features")
    _check_nan_inf(X_standard, "standard features")

    print(f"\nX_y_adaptive shape : {xy_adaptive.shape}")
    print(f"X_y_standard shape : {xy_standard.shape}")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print("\nSaving X_y_adaptive.parquet …")
    save_result(xy_adaptive, "X_y_adaptive")

    print("Saving X_y_standard.parquet …")
    save_result(xy_standard, "X_y_standard")

    # ------------------------------------------------------------------
    # 7. Correlation matrix (features vs label)
    # ------------------------------------------------------------------
    if "bin" in xy_adaptive.columns:
        print("\n" + "=" * 72)
        print("CORRELATION WITH LABEL (adaptive, Pearson)")
        print("=" * 72)
        feat_cols = config.FEATURE_NAMES
        corr = xy_adaptive[feat_cols + ["bin"]].corr()["bin"].drop("bin").sort_values()
        print(corr.round(4).to_string())

    print("\nPhase 3 (features.py) complete.")
    gc.collect()
