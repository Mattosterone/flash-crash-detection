"""
evaluation.py — Module 9: Metrics, threshold optimisation, and paper table formatting.
Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection

Public API
----------
compute_all_metrics(y_true, y_prob, y_pred)
    Compute the full metric suite from config.EVAL_METRICS.

optimize_threshold(y_true, y_prob, metric)
    Find the probability threshold that maximises a given metric.

format_table3(results)
    Build paper Table 3: main performance comparison (6 models).

format_table4(adaptive_results, standard_results)
    Build paper Table 4: adaptive vs standard labeling comparison.

format_table5(purged_results, standard_cv_results)
    Build paper Table 5: purged-embargo CV vs standard KFold (leakage).

run_labeling_comparison(...)
    Experiment RQ3: retrain best models on standard labels; compare with adaptive.

run_leakage_comparison(...)
    Experiment RQ4: retrain best models with standard KFold; compare with purged CV.
"""

import gc
import logging
import time
import warnings
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config
from src.purged_cv import PurgedEmbargoKFold
from src.utils import setup_logging

logger = setup_logging(__name__)

# Metric display names for tables and figures
METRIC_LABELS: dict[str, str] = {
    "roc_auc":     "ROC-AUC",
    "pr_auc":      "PR-AUC",
    "f1":          "F1",
    "precision":   "Precision",
    "recall":      "Recall",
    "brier_score": "Brier Score",
    "mcc":         "MCC",
}


# ======================================================================
# CORE METRIC FUNCTIONS
# ======================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all evaluation metrics from config.EVAL_METRICS.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    y_pred : np.ndarray
        Binary predictions (0 or 1) after threshold applied.

    Returns
    -------
    dict[str, float]
        Keys: roc_auc, pr_auc, f1, precision, recall, brier_score, mcc.
    """
    return {
        "roc_auc":     float(roc_auc_score(y_true, y_prob)),
        "pr_auc":      float(average_precision_score(y_true, y_prob)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    n_grid: int = config.THRESHOLD_SEARCH_GRID,
) -> float:
    """Find the probability threshold maximising the given metric.

    Leakage Rule #2 (CLAUDE.md): this function must ONLY be called on
    training-fold data. Never call it on test or held-out data.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (training fold only).
    y_prob : np.ndarray
        Predicted probabilities for positive class (training fold only).
    metric : str
        One of ``"f1"``, ``"precision"``, ``"recall"``, ``"mcc"``.
        Default ``"f1"`` (config.THRESHOLD_OPT_METRIC).
    n_grid : int
        Number of candidate thresholds in [0.01, 0.99].

    Returns
    -------
    float
        Optimal threshold in [0, 1].
    """
    thresholds = np.linspace(0.01, 0.99, n_grid)
    best_thresh, best_score = 0.5, -np.inf

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use f1/precision/recall/mcc.")
        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh


# ======================================================================
# TABLE FORMATTERS
# ======================================================================

def format_table3(results: dict[str, dict]) -> pd.DataFrame:
    """Format paper Table 3: main performance comparison across 6 models.

    Parameters
    ----------
    results : dict[str, dict]
        Keys are model keys (e.g. ``"lgbm"``); values are result dicts
        with ``mean_metrics``, ``std_metrics`` sub-dicts and timing keys.

    Returns
    -------
    pd.DataFrame
        Rows = model display names; columns = mean/std for each metric + timing.
        Index name: ``"model"``.
    """
    rows = []
    for model_key, res in results.items():
        m = res["mean_metrics"]
        s = res["std_metrics"]
        row: dict[str, Any] = {
            "model": config.MODEL_DISPLAY_NAMES.get(model_key, model_key),
        }
        for metric in config.EVAL_METRICS:
            row[f"{metric}_mean"] = round(m[metric], 4)
            row[f"{metric}_std"]  = round(s.get(metric, 0.0), 4)
        row["mean_train_time_s"]  = round(m.get("mean_train_time_s", 0), 2)
        row["total_train_time_s"] = round(m.get("total_train_time_s", 0), 2)
        row["mean_infer_time_s"]  = round(m.get("mean_infer_time_s", 0), 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df


def format_table4(
    adaptive_results: dict[str, dict],
    standard_results: dict[str, dict],
) -> pd.DataFrame:
    """Format paper Table 4: adaptive vs standard labeling comparison (RQ3).

    Each model appears twice (one row per labeling scheme). The column
    ``labeling`` distinguishes ``"adaptive"`` from ``"standard"``.

    Parameters
    ----------
    adaptive_results : dict[str, dict]
        model_key -> result dict for adaptive-labeled training runs.
    standard_results : dict[str, dict]
        model_key -> result dict for standard-labeled training runs.

    Returns
    -------
    pd.DataFrame
        Rows = (model, labeling) combinations; columns = metric mean/std.
    """
    rows = []
    all_keys = sorted(set(list(adaptive_results) + list(standard_results)))

    for model_key in all_keys:
        display = config.MODEL_DISPLAY_NAMES.get(model_key, model_key)
        for label_type, results in [("adaptive", adaptive_results),
                                     ("standard", standard_results)]:
            if model_key not in results:
                continue
            m = results[model_key]["mean_metrics"]
            s = results[model_key]["std_metrics"]
            row: dict[str, Any] = {"model": display, "labeling": label_type}
            for metric in config.EVAL_METRICS:
                row[f"{metric}_mean"] = round(m[metric], 4)
                row[f"{metric}_std"]  = round(s.get(metric, 0.0), 4)
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["model", "labeling"])
    return df


def format_table5(
    purged_results: dict[str, dict],
    standard_cv_results: dict[str, dict],
) -> pd.DataFrame:
    """Format paper Table 5: purged-embargo CV vs standard KFold (RQ4).

    Leakage-inflation is the difference (standard_cv - purged) for each metric.
    Higher standard_cv metrics confirm that standard CV inflates performance.

    Parameters
    ----------
    purged_results : dict[str, dict]
        model_key -> result dict from PurgedEmbargoKFold evaluation.
    standard_cv_results : dict[str, dict]
        model_key -> result dict from standard (non-purged) KFold evaluation.

    Returns
    -------
    pd.DataFrame
        Rows = (model, cv_type) combinations; columns = metric mean/std + inflation.
    """
    rows = []
    all_keys = sorted(set(list(purged_results) + list(standard_cv_results)))

    for model_key in all_keys:
        display = config.MODEL_DISPLAY_NAMES.get(model_key, model_key)
        for cv_type, results in [("purged_embargo", purged_results),
                                   ("standard_kfold", standard_cv_results)]:
            if model_key not in results:
                continue
            m = results[model_key]["mean_metrics"]
            s = results[model_key]["std_metrics"]
            row: dict[str, Any] = {"model": display, "cv_type": cv_type}
            for metric in config.EVAL_METRICS:
                row[f"{metric}_mean"] = round(m[metric], 4)
                row[f"{metric}_std"]  = round(s.get(metric, 0.0), 4)
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["model", "cv_type"])

    # Add inflation columns (standard_kfold - purged_embargo)
    for metric in config.EVAL_METRICS:
        col = f"{metric}_mean"
        inflation = []
        for idx in df.index:
            model, cv = idx
            purged_val = df.loc[(model, "purged_embargo"), col] \
                if (model, "purged_embargo") in df.index else np.nan
            std_val = df.loc[(model, "standard_kfold"), col] \
                if (model, "standard_kfold") in df.index else np.nan
            if cv == "standard_kfold":
                # positive = standard overstates performance vs purged
                infl = round(float(std_val - purged_val), 4) \
                    if not np.isnan(purged_val) else np.nan
            else:
                infl = np.nan
            inflation.append(infl)
        df[f"{metric}_inflation"] = inflation

    return df


# ======================================================================
# STANDARD KFOLD CV PIPELINE (for leakage experiment)
# ======================================================================

def _train_evaluate_ml_standard_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    model_name: str,
    model: Any,
    n_splits: int = config.CV_N_SPLITS,
) -> dict[str, Any]:
    """Train/evaluate one ML model using standard time-ordered KFold (no purge).

    This is intentionally LEAKY — used ONLY to quantify leakage inflation
    for RQ4 (Table 5). The fold split preserves temporal order but does NOT
    purge or embargo any events.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (18 features), DatetimeIndex sorted ascending.
    y : pd.Series
        Binary labels aligned with X.
    weights : pd.Series
        Sample uniqueness weights aligned with X.
    model_name : str
        One of ``"lgbm"``, ``"xgboost"``, ``"rf"``.
    model : sklearn-compatible classifier
        Unfitted estimator.
    n_splits : int
        Number of CV folds (time-ordered, no shuffle).

    Returns
    -------
    dict with keys: fold_metrics, mean_metrics, std_metrics, model_name.
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    X_arr = X.values.astype(np.float32)
    y_arr = y.values
    w_arr = weights.values

    # Time-ordered split: preserve temporal structure but no purge/embargo
    n = len(X_arr)
    fold_size = n // n_splits

    fold_metrics: list[dict] = []
    train_times: list[float] = []
    infer_times: list[float] = []

    for fold_num in range(1, n_splits + 1):
        test_start = (fold_num - 1) * fold_size
        test_end   = fold_num * fold_size if fold_num < n_splits else n

        train_idx = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, n),
        ])
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        w_tr = w_arr[train_idx]

        # Scale fit on train only (standard leakage rule; kept to isolate CV effect)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        fold_ratio = n_neg / n_pos if n_pos > 0 else 1.0

        fold_model = clone(model)
        if model_name == "xgboost":
            fold_model.set_params(scale_pos_weight=fold_ratio)

        t_start = time.perf_counter()
        fold_model.fit(X_tr_sc, y_tr, sample_weight=w_tr)
        train_time = time.perf_counter() - t_start
        train_times.append(train_time)

        # Threshold optimised on train (rule kept constant; only CV differs)
        proba_train = fold_model.predict_proba(X_tr_sc)[:, 1]
        opt_threshold = optimize_threshold(y_tr, proba_train)

        t_infer = time.perf_counter()
        proba_test = fold_model.predict_proba(X_te_sc)[:, 1]
        infer_time = time.perf_counter() - t_infer
        infer_times.append(infer_time)

        pred_test = (proba_test >= opt_threshold).astype(int)
        metrics = compute_all_metrics(y_te, proba_test, pred_test)
        metrics["threshold"] = opt_threshold
        metrics["fold"] = fold_num
        fold_metrics.append(metrics)

        logger.info(
            "[std_kfold/%s] Fold %d | ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f",
            model_name, fold_num,
            metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
        )

        del X_tr, X_te, X_tr_sc, X_te_sc, proba_train, proba_test
        gc.collect()

    metric_keys = list(config.EVAL_METRICS)
    mean_metrics = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in metric_keys}
    std_metrics  = {k: float(np.std([fm[k] for fm in fold_metrics], ddof=1)) for k in metric_keys}
    mean_metrics["mean_train_time_s"]  = float(np.mean(train_times))
    mean_metrics["total_train_time_s"] = float(np.sum(train_times))
    mean_metrics["mean_infer_time_s"]  = float(np.mean(infer_times))

    logger.info(
        "[std_kfold/%s] MEAN | ROC-AUC=%.4f±%.4f  PR-AUC=%.4f±%.4f",
        model_name,
        mean_metrics["roc_auc"], std_metrics["roc_auc"],
        mean_metrics["pr_auc"],  std_metrics["pr_auc"],
    )

    return {
        "fold_metrics":  fold_metrics,
        "mean_metrics":  mean_metrics,
        "std_metrics":   std_metrics,
        "model_name":    model_name,
    }


# ======================================================================
# EXPERIMENT RQ3 — LABELING COMPARISON
# ======================================================================

def run_labeling_comparison(
    X_adaptive: pd.DataFrame,
    y_adaptive: pd.Series,
    t1_adaptive: pd.Series,
    weights_adaptive: pd.Series,
    X_standard: pd.DataFrame,
    y_standard: pd.Series,
    t1_standard: pd.Series,
    weights_standard: pd.Series,
    adaptive_ml_results: dict[str, dict],
    model_subset: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """RQ3: Compare adaptive vs standard labeling for best ML models.

    Runs the same models trained with standard labels and compares against
    the already-computed adaptive results. Generates Table 4 and Figures 4-5.

    Parameters
    ----------
    X_adaptive, y_adaptive, t1_adaptive, weights_adaptive
        Feature/label/t1/weight data aligned with adaptive TBM labels.
    X_standard, y_standard, t1_standard, weights_standard
        Feature/label/t1/weight data aligned with standard TBM labels.
    adaptive_ml_results : dict[str, dict]
        Already-computed results from run_all_ml_models with adaptive labels.
    model_subset : list[str], optional
        Subset of model keys to retrain on standard labels.
        Defaults to best-performing model keys (all three ML models).

    Returns
    -------
    table4 : pd.DataFrame
        Table 4 formatted for the paper.
    fig4_path : str
        Path to Figure 4 (class distribution comparison).
    fig5_path : str
        Path to Figure 5 (metric difference adaptive − standard).
    """
    from src.models_ml import create_ml_models, train_evaluate_ml

    if model_subset is None:
        model_subset = ["lgbm", "xgboost", "rf"]

    logger.info("=" * 60)
    logger.info("RQ3: Labeling comparison — retraining on STANDARD labels")
    logger.info("=" * 60)

    # Recompute global class ratio on standard labels
    n_pos = int((y_standard == 1).sum())
    n_neg = int((y_standard == 0).sum())
    global_ratio = n_neg / n_pos if n_pos > 0 else 1.0

    outer_cv_std = PurgedEmbargoKFold(
        n_splits=config.CV_N_SPLITS,
        t1=t1_standard,
        embargo_pct=config.CV_PCT_EMBARGO,
    )

    models = create_ml_models(global_ratio)
    standard_label_results: dict[str, dict] = {}

    for name in model_subset:
        if name not in models:
            continue
        logger.info("Training %s on STANDARD labels", name)
        result = train_evaluate_ml(
            X=X_standard,
            y=y_standard,
            t1=t1_standard,
            weights=weights_standard,
            model_name=name,
            model=models[name],
            cv=outer_cv_std,
            tune=False,   # no tuning for comparison experiment (speed + fairness)
        )
        # Drop trained model objects to save memory
        result.pop("trained_models", None)
        standard_label_results[name] = result
        gc.collect()

    # ── Format Table 4 ────────────────────────────────────────────────────
    table4 = format_table4(adaptive_ml_results, standard_label_results)
    out_path = config.TABLES_DIR / "table4_labeling_comparison.csv"
    table4.to_csv(out_path)
    logger.info("Table 4 saved -> %s", out_path)

    # ── Figure 4: class distribution comparison ───────────────────────────
    fig4_path = _plot_class_distribution(y_adaptive, y_standard)

    # ── Figure 5: metric difference (adaptive − standard) ─────────────────
    fig5_path = _plot_metric_difference(
        adaptive_ml_results, standard_label_results, model_subset
    )

    return table4, fig4_path, fig5_path


# ======================================================================
# EXPERIMENT RQ4 — LEAKAGE COMPARISON
# ======================================================================

def run_leakage_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    weights: pd.Series,
    purged_ml_results: dict[str, dict],
    model_subset: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, str]:
    """RQ4: Quantify leakage inflation — standard KFold vs purged-embargo CV.

    Retrains the same ML models using standard time-ordered KFold (no purge,
    no embargo) and compares with the purged-embargo results. A higher metric
    under standard KFold confirms leakage inflation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, DatetimeIndex sorted ascending.
    y : pd.Series
        Adaptive binary labels aligned with X.
    t1 : pd.Series
        Label end-times aligned with X.
    weights : pd.Series
        Sample uniqueness weights aligned with X.
    purged_ml_results : dict[str, dict]
        Already-computed results from run_all_ml_models with purged CV.
    model_subset : list[str], optional
        Subset of model keys to retrain. Defaults to all three ML models.

    Returns
    -------
    table5 : pd.DataFrame
        Table 5 formatted for the paper.
    fig6_path : str
        Path to Figure 6 (leakage inflation visualization).
    """
    from src.models_ml import create_ml_models

    if model_subset is None:
        model_subset = ["lgbm", "xgboost", "rf"]

    logger.info("=" * 60)
    logger.info("RQ4: Leakage comparison — retraining with STANDARD KFold")
    logger.info("=" * 60)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    global_ratio = n_neg / n_pos if n_pos > 0 else 1.0

    models = create_ml_models(global_ratio)
    standard_kfold_results: dict[str, dict] = {}

    for name in model_subset:
        if name not in models:
            continue
        logger.info("Training %s with STANDARD KFold (no purge, no embargo)", name)
        result = _train_evaluate_ml_standard_kfold(
            X=X,
            y=y,
            weights=weights,
            model_name=name,
            model=models[name],
            n_splits=config.CV_N_SPLITS,
        )
        standard_kfold_results[name] = result
        gc.collect()

    # ── Format Table 5 ────────────────────────────────────────────────────
    table5 = format_table5(purged_ml_results, standard_kfold_results)
    out_path = config.TABLES_DIR / "table5_leakage_comparison.csv"
    table5.to_csv(out_path)
    logger.info("Table 5 saved -> %s", out_path)

    # ── Figure 6: leakage inflation visualization ─────────────────────────
    fig6_path = _plot_leakage_inflation(purged_ml_results, standard_kfold_results)

    return table5, fig6_path


# ======================================================================
# FIGURE GENERATORS
# ======================================================================

def _apply_plot_style() -> None:
    """Apply publication-ready matplotlib style."""
    try:
        plt.style.use(config.FIGURE_STYLE)
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "font.size":        config.FIGURE_FONT_SIZE,
        "axes.titlesize":   config.FIGURE_TITLE_SIZE,
        "axes.labelsize":   config.FIGURE_LABEL_SIZE,
        "xtick.labelsize":  config.FIGURE_TICK_SIZE,
        "ytick.labelsize":  config.FIGURE_TICK_SIZE,
        "legend.fontsize":  config.FIGURE_LEGEND_SIZE,
    })


def _plot_class_distribution(
    y_adaptive: pd.Series,
    y_standard: pd.Series,
) -> str:
    """Figure 4: class distribution under adaptive vs standard labeling.

    Saves a grouped bar chart (crash / no-crash counts) to FIGURES_DIR.

    Returns
    -------
    str
        Absolute path of the saved figure file.
    """
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (y, title) in zip(axes, [
        (y_adaptive, "Adaptive TBM"),
        (y_standard, "Standard TBM"),
    ]):
        counts = y.value_counts().sort_index()
        labels = ["No-Crash (0)", "Crash (1)"]
        values = [counts.get(0, 0), counts.get(1, 0)]
        colors = ["#56B4E9", "#E69F00"]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
        ax.set_title(title)
        ax.set_ylabel("Event Count")
        ax.set_ylim(0, max(values) * 1.25)
        for bar, val in zip(bars, values):
            pct = val / sum(values) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(
        "Figure 4: Class Distribution — Adaptive vs Standard TBM Labels",
        fontsize=config.FIGURE_TITLE_SIZE,
    )
    fig.tight_layout()

    out = config.FIGURES_DIR / f"figure4_class_distribution.{config.FIGURE_FORMAT}"
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 4 saved -> %s", out)
    return str(out)


def _plot_metric_difference(
    adaptive_results: dict[str, dict],
    standard_results: dict[str, dict],
    model_keys: list[str],
) -> str:
    """Figure 5: metric difference (adaptive − standard) per model.

    Positive values mean adaptive labeling improves the metric.
    Saves a grouped bar chart to FIGURES_DIR.

    Returns
    -------
    str
        Absolute path of the saved figure file.
    """
    _apply_plot_style()

    display_metrics = ["pr_auc", "f1", "roc_auc", "mcc"]
    metric_labels   = ["PR-AUC", "F1", "ROC-AUC", "MCC"]

    x = np.arange(len(display_metrics))
    width = 0.8 / max(len(model_keys), 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model_key in enumerate(model_keys):
        if model_key not in adaptive_results or model_key not in standard_results:
            continue
        diffs = []
        for m in display_metrics:
            adap_val = adaptive_results[model_key]["mean_metrics"].get(m, 0.0)
            std_val  = standard_results[model_key]["mean_metrics"].get(m, 0.0)
            diffs.append(round(adap_val - std_val, 4))

        display_name = config.MODEL_DISPLAY_NAMES.get(model_key, model_key)
        color = config.MODEL_COLORS.get(model_key, "#999999")
        offset = (i - len(model_keys) / 2 + 0.5) * width
        ax.bar(x + offset, diffs, width * 0.9, label=display_name, color=color,
               edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Adaptive − Standard")
    ax.set_title("Figure 5: Performance Difference (Adaptive − Standard TBM Labels)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = config.FIGURES_DIR / f"figure5_metric_difference.{config.FIGURE_FORMAT}"
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 5 saved -> %s", out)
    return str(out)


def _plot_leakage_inflation(
    purged_results: dict[str, dict],
    standard_cv_results: dict[str, dict],
) -> str:
    """Figure 6: leakage inflation — standard KFold vs purged-embargo CV.

    Shows side-by-side bars for PR-AUC and ROC-AUC with annotated inflation.
    Saves to FIGURES_DIR.

    Returns
    -------
    str
        Absolute path of the saved figure file.
    """
    _apply_plot_style()

    display_metrics = ["pr_auc", "roc_auc", "f1", "mcc"]
    metric_labels   = ["PR-AUC", "ROC-AUC", "F1", "MCC"]
    model_keys      = sorted(set(list(purged_results) + list(standard_cv_results)))

    n_models  = len(model_keys)
    n_metrics = len(display_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)

    for col, (metric, mlabel) in enumerate(zip(display_metrics, metric_labels)):
        ax = axes[col]
        purged_vals = []
        std_vals    = []
        names       = []

        for mk in model_keys:
            if mk in purged_results and mk in standard_cv_results:
                purged_vals.append(purged_results[mk]["mean_metrics"].get(metric, 0.0))
                std_vals.append(standard_cv_results[mk]["mean_metrics"].get(metric, 0.0))
                names.append(config.MODEL_DISPLAY_NAMES.get(mk, mk))

        x = np.arange(len(names))
        w = 0.35
        bars_p = ax.bar(x - w / 2, purged_vals, w, label="Purged-Embargo CV",
                        color="#0072B2", edgecolor="white")
        bars_s = ax.bar(x + w / 2, std_vals,    w, label="Standard KFold",
                        color="#E69F00", edgecolor="white")

        # Annotate inflation (std − purged)
        for xp, pv, sv in zip(x, purged_vals, std_vals):
            infl = sv - pv
            sign = "+" if infl >= 0 else ""
            ax.text(xp + w / 2, sv + 0.005, f"{sign}{infl:.3f}",
                    ha="center", va="bottom", fontsize=7, color="darkred")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(mlabel)
        ax.set_ylim(0, max(max(purged_vals + [0]), max(std_vals + [0])) * 1.3)
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Figure 6: Leakage Inflation — Standard KFold vs Purged-Embargo CV",
        fontsize=config.FIGURE_TITLE_SIZE,
    )
    fig.tight_layout()

    out = config.FIGURES_DIR / f"figure6_leakage_inflation.{config.FIGURE_FORMAT}"
    fig.savefig(out, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 6 saved -> %s", out)
    return str(out)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(config.PROJECT_ROOT))

    from src.models_ml import create_ml_models, run_all_ml_models
    from src.utils import set_reproducibility

    set_reproducibility(config.RANDOM_SEED)
    logger.info(
        "Phase 7: Evaluation + comparison experiments | LIGHTWEIGHT_MODE=%s",
        config.LIGHTWEIGHT_MODE,
    )

    # ── Load adaptive feature-label dataset ──────────────────────────────
    adap_path = config.PROCESSED_DATA_DIR / "X_y_adaptive.parquet"
    std_path  = config.PROCESSED_DATA_DIR / "X_y_standard.parquet"

    if not adap_path.exists() or not std_path.exists():
        logger.error("Missing parquet files — run Phase 3 (features.py) first.")
        sys.exit(1)

    df_adap = pd.read_parquet(adap_path)
    df_std  = pd.read_parquet(std_path)
    logger.info("Loaded X_y_adaptive shape=%s", df_adap.shape)
    logger.info("Loaded X_y_standard shape=%s", df_std.shape)

    feat_cols = config.FEATURE_NAMES

    X_adap = df_adap[feat_cols]
    y_adap = df_adap["bin"].astype(int)
    t1_adap = df_adap["t1"]
    w_adap  = df_adap["weight"]
    if not pd.api.types.is_datetime64_any_dtype(t1_adap):
        t1_adap = pd.to_datetime(t1_adap)

    X_std = df_std[feat_cols]
    y_std = df_std["bin"].astype(int)
    t1_std = df_std["t1"]
    w_std  = df_std["weight"]
    if not pd.api.types.is_datetime64_any_dtype(t1_std):
        t1_std = pd.to_datetime(t1_std)

    # ── Load or recompute adaptive ML results (purged CV) ─────────────────
    ml_results_path = config.TABLES_DIR / "table3_ml_part.csv"
    if ml_results_path.exists():
        logger.info("Found table3_ml_part.csv — rebuilding result dicts from ML re-run")

    logger.info("Running adaptive ML models (purged CV) for comparison baseline...")
    table3_ml = run_all_ml_models(X_adap, y_adap, t1_adap, w_adap, tune=False)
    logger.info("Adaptive ML (purged CV) complete.")

    # Reconstruct result dicts from the table for format_table5
    # (run_all_ml_models returns a DataFrame; we need the full result dicts
    #  — so we re-run with a wrapper that captures them)
    # For experiments we re-run train_evaluate_ml per model and collect dicts.
    from src.models_ml import create_ml_models, train_evaluate_ml

    n_pos_g = int((y_adap == 1).sum())
    n_neg_g = int((y_adap == 0).sum())
    ml_models = create_ml_models(n_neg_g / n_pos_g if n_pos_g > 0 else 1.0)
    outer_cv = PurgedEmbargoKFold(
        n_splits=config.CV_N_SPLITS,
        t1=t1_adap,
        embargo_pct=config.CV_PCT_EMBARGO,
    )

    adaptive_ml_results: dict[str, dict] = {}
    for mname, minst in ml_models.items():
        res = train_evaluate_ml(
            X=X_adap, y=y_adap, t1=t1_adap, weights=w_adap,
            model_name=mname, model=minst, cv=outer_cv, tune=False,
        )
        res.pop("trained_models", None)
        adaptive_ml_results[mname] = res
        gc.collect()

    # ── Experiment 1: RQ3 labeling comparison ────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1: RQ3 — Labeling Comparison")
    logger.info("=" * 60)
    table4, fig4_path, fig5_path = run_labeling_comparison(
        X_adaptive=X_adap,
        y_adaptive=y_adap,
        t1_adaptive=t1_adap,
        weights_adaptive=w_adap,
        X_standard=X_std,
        y_standard=y_std,
        t1_standard=t1_std,
        weights_standard=w_std,
        adaptive_ml_results=adaptive_ml_results,
    )

    print("\n" + "=" * 70)
    print("TABLE 4 — Labeling Comparison (Adaptive vs Standard TBM)")
    print("=" * 70)
    print(table4.to_string())

    # ── Experiment 2: RQ4 leakage comparison ─────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: RQ4 — Leakage Comparison")
    logger.info("=" * 60)
    table5, fig6_path = run_leakage_comparison(
        X=X_adap,
        y=y_adap,
        t1=t1_adap,
        weights=w_adap,
        purged_ml_results=adaptive_ml_results,
    )

    print("\n" + "=" * 70)
    print("TABLE 5 — Leakage Comparison (Purged-Embargo vs Standard KFold)")
    print("=" * 70)
    print(table5.to_string())

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  Table 4 -> {config.TABLES_DIR / 'table4_labeling_comparison.csv'}")
    print(f"  Table 5 -> {config.TABLES_DIR / 'table5_leakage_comparison.csv'}")
    print(f"  Figure 4 -> {fig4_path}")
    print(f"  Figure 5 -> {fig5_path}")
    print(f"  Figure 6 -> {fig6_path}")

    # Sanity check: warn if standard KFold is NOT better than purged CV
    print("\n" + "=" * 70)
    print("LEAKAGE SANITY CHECK")
    print("=" * 70)
    for mname in ["lgbm", "xgboost", "rf"]:
        if mname not in adaptive_ml_results:
            continue
        purged_pr = adaptive_ml_results[mname]["mean_metrics"].get("pr_auc", 0)
        std_cv_row = table5.xs("standard_kfold", level="cv_type") \
            if "standard_kfold" in table5.index.get_level_values("cv_type") else None
        disp = config.MODEL_DISPLAY_NAMES.get(mname, mname)
        if std_cv_row is not None and disp in std_cv_row.index:
            std_pr = std_cv_row.loc[disp, "pr_auc_mean"]
            direction = "EXPECTED (leakage confirmed)" if std_pr > purged_pr \
                else "WARNING: standard CV WORSE than purged — check folds"
            print(f"  {disp}: purged={purged_pr:.4f}  std_kfold={std_pr:.4f}  {direction}")
