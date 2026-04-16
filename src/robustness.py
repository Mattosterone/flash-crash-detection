"""
robustness.py — Module 11: Robustness Checks and Computational Feasibility.
Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection

Implements Phase 9 of the thesis pipeline:
  - Table 7: Computational feasibility (training time, inference time,
             parameter counts, relative resource usage)
  - Table 8: Robustness checks across 8 experimental settings
  - Figure 10: Sensitivity plot (recall / PR-AUC / MCC vs. setting)

Robustness settings (all evaluated on best model = Random Forest):
  1. Baseline       : horizon=60, pt=1.5, sl=1.0, embargo=0.01, features=all
  2. Alt horizon 30 : horizon=30, same barriers/embargo, features=all
  3. Alt horizon 90 : horizon=90, same barriers/embargo, features=all
  4. Narrow barriers: horizon=60, pt=1.0, sl=0.5, embargo=0.01, features=all
  5. Wide barriers  : horizon=60, pt=2.0, sl=1.5, embargo=0.01, features=all
  6. Top-5 features : baseline params, features=top-5 by RF importance
  7. Embargo 0.5%   : baseline params, embargo_pct=0.005
  8. Embargo 2.0%   : baseline params, embargo_pct=0.02

CV uses 3-fold PurgedEmbargoKFold in all robustness checks
(lighter than the 5-fold used for main results; consistent within Table 8).
No nested hyperparameter tuning — default RF params throughout.

Public API
----------
run_robustness_phase(X_base, y_base, t1_base, weights_base, df_clean, cusum_events)
    Orchestrator: runs all checks, saves Table 7, Table 8, Figure 10.
"""

import gc
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import config
from src.labeling import apply_standard_tbm
from src.features import get_feature_matrix
from src.sample_weights import compute_sample_weights
from src.purged_cv import PurgedEmbargoKFold
from src.utils import setup_logging

logger = setup_logging(__name__)

# Number of CV folds for robustness checks (3 for speed; noted in paper)
_ROBUST_CV_SPLITS: int = 3


# ======================================================================
# ROBUSTNESS SETTING DEFINITIONS
# ======================================================================

ROBUSTNESS_SETTINGS: list[dict] = [
    {
        "name": "Baseline",
        "horizon": config.TBM_HORIZON_BARS,    # 60
        "pt":      config.TBM_PROFIT_TAKING,   # 1.5
        "sl":      config.TBM_STOP_LOSS,       # 1.0
        "embargo_pct": config.CV_PCT_EMBARGO,  # 0.01
        "feature_subset": "all",
        "relabel": False,   # use pre-computed adaptive labels
    },
    {
        "name": "Horizon 30min",
        "horizon": 30,
        "pt":      config.TBM_PROFIT_TAKING,
        "sl":      config.TBM_STOP_LOSS,
        "embargo_pct": config.CV_PCT_EMBARGO,
        "feature_subset": "all",
        "relabel": True,
    },
    {
        "name": "Horizon 90min",
        "horizon": 90,
        "pt":      config.TBM_PROFIT_TAKING,
        "sl":      config.TBM_STOP_LOSS,
        "embargo_pct": config.CV_PCT_EMBARGO,
        "feature_subset": "all",
        "relabel": True,
    },
    {
        "name": "Narrow Barriers",
        "horizon": config.TBM_HORIZON_BARS,
        "pt":      1.0,
        "sl":      0.5,
        "embargo_pct": config.CV_PCT_EMBARGO,
        "feature_subset": "all",
        "relabel": True,
    },
    {
        "name": "Wide Barriers",
        "horizon": config.TBM_HORIZON_BARS,
        "pt":      2.0,
        "sl":      1.5,
        "embargo_pct": config.CV_PCT_EMBARGO,
        "feature_subset": "all",
        "relabel": True,
    },
    {
        "name": "Top-5 Features",
        "horizon": config.TBM_HORIZON_BARS,
        "pt":      config.TBM_PROFIT_TAKING,
        "sl":      config.TBM_STOP_LOSS,
        "embargo_pct": config.CV_PCT_EMBARGO,
        "feature_subset": "top5",
        "relabel": False,
    },
    {
        "name": "Embargo 0.5%",
        "horizon": config.TBM_HORIZON_BARS,
        "pt":      config.TBM_PROFIT_TAKING,
        "sl":      config.TBM_STOP_LOSS,
        "embargo_pct": 0.005,
        "feature_subset": "all",
        "relabel": False,
    },
    {
        "name": "Embargo 2.0%",
        "horizon": config.TBM_HORIZON_BARS,
        "pt":      config.TBM_PROFIT_TAKING,
        "sl":      config.TBM_STOP_LOSS,
        "embargo_pct": 0.02,
        "feature_subset": "all",
        "relabel": False,
    },
]


# ======================================================================
# HELPERS
# ======================================================================

def _load_rf_best() -> Optional[RandomForestClassifier]:
    """Load the best Random Forest model saved during Phase 5.

    Returns
    -------
    RandomForestClassifier or None if the file is not found.
    """
    rf_path = config.PROCESSED_DATA_DIR / "models" / "rf_best.pkl"
    if not rf_path.exists():
        logger.warning("rf_best.pkl not found at %s — top-5 features unavailable", rf_path)
        return None
    with open(rf_path, "rb") as fh:
        model = pickle.load(fh)
    logger.info("Loaded rf_best.pkl from %s", rf_path)
    return model


def _get_top5_features(rf_model: Optional[RandomForestClassifier]) -> list[str]:
    """Return the top-5 feature names by RF Gini importance.

    Falls back to the first 5 features in config.FEATURE_NAMES if the
    model is not available.

    Parameters
    ----------
    rf_model : RandomForestClassifier or None

    Returns
    -------
    list of 5 feature names.
    """
    if rf_model is None or not hasattr(rf_model, "feature_importances_"):
        logger.warning(
            "RF importance unavailable; using first 5 features as top-5 proxy."
        )
        return config.FEATURE_NAMES[:5]

    importances = rf_model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    top5 = [config.FEATURE_NAMES[i] for i in idx_sorted[:5]]
    logger.info("Top-5 features by RF importance: %s", top5)
    return top5


def _optimize_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold that maximises F1 on the supplied data."""
    thresholds = np.linspace(0.01, 0.99, config.THRESHOLD_SEARCH_GRID)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
) -> dict[str, float]:
    """Full metric suite (same as models_ml._compute_metrics)."""
    return {
        "roc_auc":     roc_auc_score(y_true, proba),
        "pr_auc":      average_precision_score(y_true, proba),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, proba),
        "mcc":         matthews_corrcoef(y_true, y_pred),
    }


# ======================================================================
# RELABELING + FEATURE ALIGNMENT
# ======================================================================

def _build_dataset_for_setting(
    setting: dict,
    X_base: pd.DataFrame,
    y_base: pd.Series,
    t1_base: pd.Series,
    weights_base: pd.Series,
    df_clean: pd.DataFrame,
    cusum_events: pd.DatetimeIndex,
    df_features: pd.DataFrame,
    top5_features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return (X, y, t1, weights) for a given robustness setting.

    If ``setting["relabel"]`` is False, the baseline adaptive labels are
    used (with optional feature subsetting or embargo change).
    If ``setting["relabel"]`` is True, ``apply_standard_tbm`` is called
    with the setting's pt / sl / horizon to generate new labels, then
    features are re-aligned.

    Parameters
    ----------
    setting : dict
        One entry from ROBUSTNESS_SETTINGS.
    X_base, y_base, t1_base, weights_base
        Pre-computed baseline (adaptive labels, all features).
    df_clean : pd.DataFrame
        Full cleaned OHLCV DataFrame (needed for relabeling).
    cusum_events : pd.DatetimeIndex
        CUSUM event timestamps.
    df_features : pd.DataFrame
        Full-bar feature DataFrame (engineered features on df_clean).
    top5_features : list[str]
        Top-5 feature names (from RF importance).

    Returns
    -------
    tuple (X, y, t1, weights) ready for CV training.
    """
    if setting["relabel"]:
        # Compute volatility from df_clean (same EWM as Phase 2)
        volatility = (
            df_clean["log_return"]
            .ewm(span=config.VOLATILITY_EWM_SPAN, min_periods=config.VOLATILITY_EWM_SPAN)
            .std()
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = apply_standard_tbm(
                close=df_clean["close"],
                events=cusum_events,
                volatility=volatility,
                pt=setting["pt"],
                sl=setting["sl"],
                max_horizon=setting["horizon"],
            )

        logger.info(
            "[%s] Re-labeled: %d events, crash_rate=%.3f",
            setting["name"], len(labels), labels["bin"].mean(),
        )

        # Align features to new label index
        X_new = get_feature_matrix(df_features, labels)
        y_new = labels["bin"].reindex(X_new.index)
        t1_new = labels["t1"].reindex(X_new.index)

        # Drop rows with NaN in y or t1
        valid = y_new.notna() & t1_new.notna()
        X_new = X_new.loc[valid]
        y_new = y_new.loc[valid]
        t1_new = t1_new.loc[valid]

        # Recompute sample weights for new label set
        weights_new = compute_sample_weights(
            event_starts=X_new.index,
            event_ends=t1_new,
            close_index=df_clean.index,
        )

        X_out, y_out, t1_out, w_out = X_new, y_new, t1_new, weights_new
        del labels, volatility
        gc.collect()
    else:
        X_out, y_out, t1_out, w_out = X_base.copy(), y_base.copy(), t1_base.copy(), weights_base.copy()

    # Apply feature subset
    if setting["feature_subset"] == "top5":
        X_out = X_out[top5_features]

    return X_out, y_out, t1_out, w_out


# ======================================================================
# RF TRAINING LOOP (lightweight, no nested tuning)
# ======================================================================

def _train_rf_cv(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    weights: pd.Series,
    embargo_pct: float,
    n_splits: int = _ROBUST_CV_SPLITS,
) -> dict[str, float]:
    """Train Random Forest with PurgedEmbargoKFold, return mean metrics.

    Uses default RF params from config (no nested hyperparameter tuning).
    Leakage rules enforced:
      1. Scaler fit on train only.
      2. Threshold optimised on train predictions only.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    t1 : pd.Series
    weights : pd.Series
    embargo_pct : float
    n_splits : int

    Returns
    -------
    dict with mean roc_auc, pr_auc, f1, precision, recall, brier_score,
    mcc, mean_train_time_s, mean_infer_time_s.
    """
    cv = PurgedEmbargoKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)

    X_arr = X.values.astype(np.float32)
    y_arr = y.values
    w_arr = weights.values

    base_model = RandomForestClassifier(**config.RF_PARAMS)

    fold_metrics: list[dict] = []
    train_times: list[float] = []
    infer_times: list[float] = []

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        w_tr = w_arr[train_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            logger.warning("Fold %d skipped — single class in train or test.", fold_num)
            continue

        # Leakage rule 1: scale fit on train
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # Train
        model = clone(base_model)
        t0 = time.perf_counter()
        model.fit(X_tr_sc, y_tr, sample_weight=w_tr)
        train_times.append(time.perf_counter() - t0)

        # Leakage rule 2: threshold on train predictions
        proba_tr = model.predict_proba(X_tr_sc)[:, 1]
        threshold = _optimize_threshold(y_tr, proba_tr)

        # Inference on test
        t_infer = time.perf_counter()
        proba_te = model.predict_proba(X_te_sc)[:, 1]
        infer_times.append(time.perf_counter() - t_infer)

        pred_te = (proba_te >= threshold).astype(int)
        m = _compute_metrics(y_te, pred_te, proba_te)
        m["fold"] = fold_num
        fold_metrics.append(m)

        logger.debug(
            "Fold %d | PR-AUC=%.4f  Recall=%.4f  MCC=%.4f  train=%.1fs",
            fold_num, m["pr_auc"], m["recall"], m["mcc"],
            train_times[-1],
        )

        del X_tr, X_te, X_tr_sc, X_te_sc
        gc.collect()

    if not fold_metrics:
        logger.error("All folds skipped — returning zeros.")
        return {k: 0.0 for k in config.EVAL_METRICS + ["mean_train_time_s", "mean_infer_time_s"]}

    metric_keys = config.EVAL_METRICS
    result = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in metric_keys}
    result["mean_train_time_s"] = float(np.mean(train_times))
    result["mean_infer_time_s"] = float(np.mean(infer_times))
    return result


# ======================================================================
# TABLE 7 — COMPUTATIONAL FEASIBILITY
# ======================================================================

def _count_dl_params(input_size: int, hidden_size: int, num_layers: int) -> dict[str, int]:
    """Compute parameter counts for RNN, LSTM, GRU analytically.

    For a single recurrent layer:
      RNN  : (input * hidden) + (hidden * hidden) + 2*hidden
      LSTM : 4 * ((input * hidden) + (hidden * hidden) + 2*hidden)
      GRU  : 3 * ((input * hidden) + (hidden * hidden) + 2*hidden)

    Parameters follow PyTorch defaults (bias_ih + bias_hh per gate).
    """
    def _layer_params(rnn_type: str, in_sz: int, h_sz: int) -> int:
        gate_multiplier = {"RNN": 1, "LSTM": 4, "GRU": 3}[rnn_type]
        return gate_multiplier * (in_sz * h_sz + h_sz * h_sz + 2 * h_sz)

    counts: dict[str, int] = {}
    for rnn_type in ("RNN", "LSTM", "GRU"):
        total = 0
        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size
            total += _layer_params(rnn_type, in_sz, hidden_size)
        total += hidden_size + 1  # FC: weight + bias
        counts[rnn_type] = total

    return counts


def compile_table7_computational(
    ml_csv_path: Optional[Path] = None,
    dl_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build Table 7: computational feasibility summary.

    Compiles training/inference timing from saved CSV files (Phases 5–6),
    adds parameter count estimates, and computes relative resource usage
    (normalised to LightGBM = 1.0).

    Parameters
    ----------
    ml_csv_path : Path, optional
        Path to table3_ml_part.csv.  Defaults to config.TABLES_DIR / "table3_ml_part.csv".
    dl_csv_path : Path, optional
        Path to table3_dl_part.csv.  Defaults to config.TABLES_DIR / "table3_dl_part.csv".

    Returns
    -------
    pd.DataFrame
        Table 7 with columns:
          model, n_params, mean_train_time_s, mean_infer_time_s,
          total_train_time_s, relative_train_time, notes
    """
    if ml_csv_path is None:
        ml_csv_path = config.TABLES_DIR / "table3_ml_part.csv"
    if dl_csv_path is None:
        dl_csv_path = config.TABLES_DIR / "table3_dl_part.csv"

    # ── ML timing ────────────────────────────────────────────────────────
    ml_df = pd.read_csv(ml_csv_path)
    timing_rows: list[dict] = []

    for _, row in ml_df.iterrows():
        timing_rows.append({
            "model":             row["model"],
            "n_params":          _estimate_ml_params(row["model"]),
            "mean_train_time_s": row["mean_train_time_s"],
            "total_train_time_s": row["total_train_time_s"],
            "mean_infer_time_s": row["mean_infer_time_s"],
            "notes":             "5-fold purged CV (lightweight)",
        })

    # ── DL timing ────────────────────────────────────────────────────────
    effective_lookback = (
        config.DL_LOOKBACK if not config.LIGHTWEIGHT_MODE else 10
    )
    dl_param_counts = _count_dl_params(
        input_size=config.N_FEATURES,
        hidden_size=config.DL_HIDDEN_SIZE,
        num_layers=config.DL_NUM_LAYERS,
    )

    if dl_csv_path.exists():
        dl_df = pd.read_csv(dl_csv_path)
        for _, row in dl_df.iterrows():
            arch = row["model"]
            rnn_key = arch.upper()  # "RNN", "LSTM", "GRU"
            timing_rows.append({
                "model":             arch,
                "n_params":          dl_param_counts.get(rnn_key, "N/A"),
                "mean_train_time_s": row["mean_train_time_s"],
                "total_train_time_s": row["total_train_time_s"],
                "mean_infer_time_s": row["mean_infer_time_s"],
                "notes":             f"5-fold purged CV (lightweight, lookback={effective_lookback})",
            })
    else:
        logger.warning(
            "table3_dl_part.csv not found (%s) — DL timing shown as N/A. "
            "Run Phase 6 (run_all_dl_models) to populate.",
            dl_csv_path,
        )
        for arch in ("RNN", "LSTM", "GRU"):
            timing_rows.append({
                "model":             arch,
                "n_params":          dl_param_counts[arch],
                "mean_train_time_s": float("nan"),
                "total_train_time_s": float("nan"),
                "mean_infer_time_s": float("nan"),
                "notes":             "timing N/A — run Phase 6 for data",
            })

    table7 = pd.DataFrame(timing_rows)

    # ── Relative training time (LightGBM = 1.0) ─────────────────────────
    lgbm_time = table7.loc[table7["model"] == "LightGBM", "mean_train_time_s"]
    if len(lgbm_time) > 0 and not pd.isna(lgbm_time.iloc[0]) and lgbm_time.iloc[0] > 0:
        ref = lgbm_time.iloc[0]
        table7["relative_train_time"] = table7["mean_train_time_s"].apply(
            lambda t: round(t / ref, 2) if pd.notna(t) else float("nan")
        )
    else:
        table7["relative_train_time"] = float("nan")

    # Round numeric columns
    for col in ["mean_train_time_s", "total_train_time_s", "mean_infer_time_s"]:
        table7[col] = table7[col].round(4)

    logger.info("Table 7 compiled (%d rows)", len(table7))
    return table7


def _estimate_ml_params(model_name: str) -> str:
    """Return a human-readable parameter count estimate for ML models.

    These are approximate node/leaf counts, not weight vectors.
    """
    estimates = {
        "LightGBM":     "~15,500 leaves (500 trees × ~31 leaves)",
        "XGBoost":      "~15,500 leaves (500 trees × ~31 leaves)",
        "Random Forest": "~256,000 nodes (500 trees, max_depth=10)",
    }
    return estimates.get(model_name, "N/A")


# ======================================================================
# TABLE 8 — ROBUSTNESS CHECKS
# ======================================================================

def run_all_robustness_checks(
    X_base: pd.DataFrame,
    y_base: pd.Series,
    t1_base: pd.Series,
    weights_base: pd.Series,
    df_clean: pd.DataFrame,
    cusum_events: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Run all 8 robustness settings on the best model (Random Forest).

    For each setting:
      1. Build dataset (relabel if needed, subset features if needed).
      2. Train RF with _ROBUST_CV_SPLITS-fold PurgedEmbargoKFold.
      3. Collect mean recall, PR-AUC, MCC.

    Parameters
    ----------
    X_base : pd.DataFrame
        Baseline feature matrix (adaptive labels, all 18 features).
    y_base : pd.Series
        Baseline labels.
    t1_base : pd.Series
        Baseline label end-times.
    weights_base : pd.Series
        Baseline sample weights.
    df_clean : pd.DataFrame
        Full cleaned OHLCV DataFrame with log_return.
    cusum_events : pd.DatetimeIndex
        CUSUM event timestamps.

    Returns
    -------
    pd.DataFrame
        Table 8 with columns: setting, n_events, crash_rate,
        horizon, pt, sl, embargo_pct, feature_subset,
        recall_mean, pr_auc_mean, mcc_mean, roc_auc_mean, f1_mean.
    """
    # Pre-compute full-bar features once (shared across relabeling settings)
    logger.info("Pre-computing full-bar feature DataFrame for relabeling...")
    from src.features import engineer_features
    df_features = engineer_features(df_clean)

    # Load top-5 features
    rf_model = _load_rf_best()
    top5_features = _get_top5_features(rf_model)

    rows: list[dict] = []

    for setting in ROBUSTNESS_SETTINGS:
        logger.info("=" * 60)
        logger.info("Robustness check: %s", setting["name"])
        logger.info("=" * 60)

        # Build dataset
        X, y, t1, w = _build_dataset_for_setting(
            setting=setting,
            X_base=X_base,
            y_base=y_base,
            t1_base=t1_base,
            weights_base=weights_base,
            df_clean=df_clean,
            cusum_events=cusum_events,
            df_features=df_features,
            top5_features=top5_features,
        )

        n_events = len(y)
        crash_rate = float(y.mean())
        n_features_used = X.shape[1]
        feature_label = (
            f"top5: {top5_features}"
            if setting["feature_subset"] == "top5"
            else "all (18)"
        )

        logger.info(
            "[%s] Dataset ready: n=%d, crash_rate=%.3f, features=%d, embargo=%.3f",
            setting["name"], n_events, crash_rate, n_features_used,
            setting["embargo_pct"],
        )

        # Train & evaluate
        metrics = _train_rf_cv(
            X=X,
            y=y,
            t1=t1,
            weights=w,
            embargo_pct=setting["embargo_pct"],
            n_splits=_ROBUST_CV_SPLITS,
        )

        row = {
            "setting":        setting["name"],
            "n_events":       n_events,
            "crash_rate":     round(crash_rate, 4),
            "horizon_bars":   setting["horizon"],
            "pt_multiplier":  setting["pt"],
            "sl_multiplier":  setting["sl"],
            "embargo_pct":    setting["embargo_pct"],
            "feature_subset": feature_label,
            "roc_auc":        round(metrics["roc_auc"], 4),
            "pr_auc":         round(metrics["pr_auc"], 4),
            "f1":             round(metrics["f1"], 4),
            "recall":         round(metrics["recall"], 4),
            "precision":      round(metrics["precision"], 4),
            "mcc":            round(metrics["mcc"], 4),
            "brier_score":    round(metrics["brier_score"], 4),
        }
        rows.append(row)

        logger.info(
            "[%s] Result: Recall=%.4f  PR-AUC=%.4f  MCC=%.4f",
            setting["name"], row["recall"], row["pr_auc"], row["mcc"],
        )

        # Release memory
        del X, y, t1, w
        gc.collect()

    del df_features
    gc.collect()

    return pd.DataFrame(rows)


# ======================================================================
# FIGURE 10 — SENSITIVITY PLOT
# ======================================================================

def build_figure10_sensitivity(table8: pd.DataFrame) -> None:
    """Build Figure 10: sensitivity of recall / PR-AUC / MCC across settings.

    Creates a multi-panel bar chart showing how each robustness metric
    changes relative to the Baseline setting.  Saves as PDF and PNG.

    Parameters
    ----------
    table8 : pd.DataFrame
        Output of run_all_robustness_checks.
    """
    try:
        plt.style.use(config.FIGURE_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")

    metrics = ["pr_auc", "recall", "mcc"]
    metric_labels = {"pr_auc": "PR-AUC", "recall": "Recall", "mcc": "MCC"}
    colors = {"pr_auc": "#0072B2", "recall": "#009E73", "mcc": "#E69F00"}

    settings_ordered = table8["setting"].tolist()
    x = np.arange(len(settings_ordered))
    n_metrics = len(metrics)
    bar_width = 0.25
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_width

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, metric in enumerate(metrics):
        values = table8[metric].values.astype(float)
        bars = ax.bar(
            x + offsets[i],
            values,
            width=bar_width,
            label=metric_labels[metric],
            color=colors[metric],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Annotate each bar with its value
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="black",
            )

    # Baseline reference lines
    baseline_row = table8[table8["setting"] == "Baseline"].iloc[0]
    for metric in metrics:
        ax.axhline(
            baseline_row[metric],
            color=colors[metric],
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        settings_ordered,
        rotation=30,
        ha="right",
        fontsize=config.FIGURE_TICK_SIZE,
    )
    ax.set_ylabel("Metric Value", fontsize=config.FIGURE_LABEL_SIZE)
    ax.set_xlabel("Robustness Setting", fontsize=config.FIGURE_LABEL_SIZE)
    ax.set_title(
        "Figure 10: Sensitivity of RF Performance Across Robustness Settings\n"
        "(dashed lines = baseline values)",
        fontsize=config.FIGURE_TITLE_SIZE,
        pad=12,
    )
    ax.legend(fontsize=config.FIGURE_LEGEND_SIZE, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = config.FIGURES_DIR / f"figure10_sensitivity.{ext}"
        fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
        logger.info("Figure 10 saved -> %s", out)

    plt.close(fig)


# ======================================================================
# ORCHESTRATOR
# ======================================================================

def run_robustness_phase(
    X_base: pd.DataFrame,
    y_base: pd.Series,
    t1_base: pd.Series,
    weights_base: pd.Series,
    df_clean: pd.DataFrame,
    cusum_events: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Run all Phase 9 analyses and save outputs.

    Steps:
      1. Compile Table 7 (computational feasibility).
      2. Run all 8 robustness settings -> Table 8.
      3. Build Figure 10 (sensitivity plot).
      4. Save Table 7 and Table 8 as CSV.

    Parameters
    ----------
    X_base : pd.DataFrame
        Feature matrix from adaptive labeling run (Phase 3).
    y_base : pd.Series
        Adaptive labels.
    t1_base : pd.Series
        Label end-times.
    weights_base : pd.Series
        Sample weights.
    df_clean : pd.DataFrame
        Cleaned OHLCV data.
    cusum_events : pd.DatetimeIndex
        CUSUM event timestamps.

    Returns
    -------
    dict with keys ``"table7"`` and ``"table8"`` (DataFrames).
    """
    # ── Table 7 ──────────────────────────────────────────────────────────
    logger.info("Compiling Table 7 (computational feasibility)...")
    table7 = compile_table7_computational()
    t7_path = config.TABLES_DIR / "table7_computational.csv"
    table7.to_csv(t7_path, index=False)
    logger.info("Table 7 saved -> %s", t7_path)

    # Print Table 7
    print("\n" + "=" * 70)
    print("TABLE 7 — COMPUTATIONAL FEASIBILITY")
    print("=" * 70)
    print(table7.to_string(index=False))
    print("=" * 70)

    # ── Table 8 ──────────────────────────────────────────────────────────
    logger.info("Running robustness checks (Table 8)...")
    table8 = run_all_robustness_checks(
        X_base=X_base,
        y_base=y_base,
        t1_base=t1_base,
        weights_base=weights_base,
        df_clean=df_clean,
        cusum_events=cusum_events,
    )
    t8_path = config.TABLES_DIR / "table8_robustness.csv"
    table8.to_csv(t8_path, index=False)
    logger.info("Table 8 saved -> %s", t8_path)

    # Print Table 8
    print("\n" + "=" * 70)
    print("TABLE 8 — ROBUSTNESS CHECKS (Random Forest, 3-fold purged CV)")
    print("=" * 70)
    display_cols = ["setting", "n_events", "crash_rate", "horizon_bars",
                    "pt_multiplier", "sl_multiplier", "embargo_pct",
                    "recall", "pr_auc", "mcc"]
    print(table8[display_cols].to_string(index=False))
    print("=" * 70)

    # ── Figure 10 ────────────────────────────────────────────────────────
    logger.info("Building Figure 10 (sensitivity plot)...")
    build_figure10_sensitivity(table8)

    logger.info("Phase 9 complete.")
    return {"table7": table7, "table8": table8}


# ======================================================================
# STANDALONE ENTRY POINT (for direct script execution)
# ======================================================================

def main() -> None:
    """Load pre-computed data and run Phase 9 robustness analysis.

    Loads:
      - data/processed/X_y_adaptive.parquet  -> X_base, y_base
      - data/processed/labels_adaptive.parquet -> t1_base
      - data/processed/df_clean.parquet       -> df_clean
      - data/processed/cusum_events.parquet   -> cusum_events

    Recomputes sample weights (fast, deterministic).
    """
    logger.info("Loading pre-computed data for Phase 9...")

    # ── X, y (adaptive) ──────────────────────────────────────────────────
    xy_path = config.PROCESSED_DATA_DIR / "X_y_adaptive.parquet"
    if not xy_path.exists():
        raise FileNotFoundError(
            f"X_y_adaptive.parquet not found at {xy_path}. "
            "Run Phase 3 (features.py) first."
        )
    xy = pd.read_parquet(xy_path)
    feature_cols = [c for c in config.FEATURE_NAMES if c in xy.columns]
    X_base = xy[feature_cols].copy()
    y_base = xy["bin"].copy()
    logger.info("Loaded X_y_adaptive: %d events, %d features", len(X_base), len(feature_cols))

    # ── t1 (label end-times from adaptive labels) ────────────────────────
    labels_path = config.PROCESSED_DATA_DIR / "labels_adaptive.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels_adaptive.parquet not found at {labels_path}. "
            "Run Phase 2 (labeling.py) first."
        )
    labels_adaptive = pd.read_parquet(labels_path)
    t1_base = labels_adaptive["t1"].reindex(X_base.index)

    # ── df_clean ─────────────────────────────────────────────────────────
    clean_path = config.PROCESSED_DATA_DIR / "df_clean.parquet"
    df_clean = pd.read_parquet(clean_path)
    logger.info("Loaded df_clean: %d bars", len(df_clean))

    # ── CUSUM events ─────────────────────────────────────────────────────
    cusum_path = config.PROCESSED_DATA_DIR / "cusum_events.parquet"
    cusum_df = pd.read_parquet(cusum_path)
    cusum_events = cusum_df.index
    logger.info("Loaded CUSUM events: %d", len(cusum_events))

    # ── Sample weights ────────────────────────────────────────────────────
    logger.info("Computing sample weights for baseline...")
    weights_base = compute_sample_weights(
        event_starts=X_base.index,
        event_ends=t1_base,
        close_index=df_clean.index,
    )

    # ── Run Phase 9 ──────────────────────────────────────────────────────
    run_robustness_phase(
        X_base=X_base,
        y_base=y_base,
        t1_base=t1_base,
        weights_base=weights_base,
        df_clean=df_clean,
        cusum_events=cusum_events,
    )


if __name__ == "__main__":
    main()
