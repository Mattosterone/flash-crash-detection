"""
models_ml.py — Module 7: ML model training under Purged-Embargo CV.
Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection

Implements leakage-aware training pipelines for three tree-based models:
  - LightGBM (lgbm)
  - XGBoost  (xgboost)
  - Random Forest (rf)

Leakage safeguards per fold (non-negotiable, see CLAUDE.md):
  1. Feature scaling fit on TRAIN ONLY; test transformed with train scaler.
  2. Classification threshold optimized on TRAIN ONLY (max F1 over grid).
  3. XGBoost scale_pos_weight computed from actual training fold class ratio.
  4. Nested CV for hyperparameter tuning: inner PurgedEmbargoKFold within
     each outer training fold; tuning never touches test fold.

Public API
----------
create_ml_models(class_ratio)
    Build initial model instances from config.

train_evaluate_ml(X, y, t1, weights, model_name, model, cv, tune)
    Full purged-CV pipeline for one model; returns per-fold + aggregate metrics.

run_all_ml_models(X, y, t1, weights, tune)
    Train & evaluate all 3 models; returns DataFrame for Table 3.
"""

import gc
import pickle
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
from src.purged_cv import PurgedEmbargoKFold
from src.utils import setup_logging

logger = setup_logging(__name__)


# ======================================================================
# MODEL FACTORY
# ======================================================================

def create_ml_models(class_ratio: float) -> dict[str, Any]:
    """Instantiate all three ML classifiers from config parameters.

    Parameters
    ----------
    class_ratio : float
        Global n_negative / n_positive ratio used as the initial
        scale_pos_weight for XGBoost.  This value is overridden per
        fold inside ``train_evaluate_ml`` using the actual fold ratio.

    Returns
    -------
    dict
        Keys: ``"lgbm"``, ``"xgboost"``, ``"rf"``.
        Values: unfitted sklearn-compatible classifier instances.
    """
    lgbm = LGBMClassifier(**config.LGBM_PARAMS)

    # XGBoost: drop deprecated use_label_encoder (removed in XGBoost >= 1.6)
    xgb_params = {k: v for k, v in config.XGBOOST_PARAMS.items()
                  if k != "use_label_encoder"}
    xgb_params["scale_pos_weight"] = class_ratio
    xgb = XGBClassifier(**xgb_params)

    rf = RandomForestClassifier(**config.RF_PARAMS)

    logger.info(
        "Created ML models | lgbm / xgboost / rf | initial class_ratio=%.3f",
        class_ratio,
    )
    return {"lgbm": lgbm, "xgboost": xgb, "rf": rf}


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _optimize_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_grid: int = config.THRESHOLD_SEARCH_GRID,
) -> float:
    """Find the probability threshold that maximises F1 on the given data.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    proba : np.ndarray
        Predicted probabilities for the positive class.
    n_grid : int
        Number of candidate thresholds uniformly spaced in (0, 1).

    Returns
    -------
    float
        Threshold in [0, 1] that maximised F1 on the supplied data.
    """
    thresholds = np.linspace(0.01, 0.99, n_grid)
    best_thresh, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return float(best_thresh)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
) -> dict[str, float]:
    """Compute all evaluation metrics from config.EVAL_METRICS.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Binary predictions (after threshold applied).
    proba : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Keys match config.EVAL_METRICS:
        roc_auc, pr_auc, f1, precision, recall, brier_score, mcc.
    """
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
# PER-MODEL CV TRAINING PIPELINE
# ======================================================================

def train_evaluate_ml(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    weights: pd.Series,
    model_name: str,
    model: Any,
    cv: PurgedEmbargoKFold,
    tune: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one ML model under Purged-Embargo CV.

    For each outer CV fold the pipeline is:
      1. Split via ``cv`` (PurgedEmbargoKFold).
      2. Fit StandardScaler on train, transform both train and test.
      3. If ``tune``: nested RandomizedSearchCV on train fold with inner
         PurgedEmbargoKFold to find best hyperparameters.
      4. For XGBoost: compute scale_pos_weight from fold train class ratio.
      5. Fit final model on train fold with sample_weight.
      6. Predict probabilities on test fold.
      7. Optimise threshold (max F1) on TRAIN predictions only.
      8. Apply threshold to test predictions.
      9. Compute all metrics on test fold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (18 features), DatetimeIndex.
    y : pd.Series
        Binary labels aligned with X.
    t1 : pd.Series
        Label end-times (barrier hit times) aligned with X.
    weights : pd.Series
        Sample uniqueness weights aligned with X.
    model_name : str
        One of ``"lgbm"``, ``"xgboost"``, ``"rf"``.
    model : sklearn-compatible classifier
        Unfitted estimator (will be cloned per fold).
    cv : PurgedEmbargoKFold
        Pre-built CV splitter with ``t1`` aligned to ``X``.
    tune : bool
        If True, run inner RandomizedSearchCV to find best hyperparameters
        before final fold training.  Default False.

    Returns
    -------
    dict with keys:
        fold_metrics   : list[dict]  — per-fold metric dicts
        mean_metrics   : dict        — mean across folds
        std_metrics    : dict        — std  across folds
        predictions    : pd.DataFrame — test-fold index + proba + pred + label
        best_params    : list[dict]  — best params per fold (None if tune=False)
        train_times    : list[float] — seconds per fold
        infer_times    : list[float] — seconds per fold
        model_name     : str
        trained_models : list        — fitted models (one per fold)
    """
    X_arr = X.values.astype(np.float32)
    y_arr = y.values
    w_arr = weights.values

    fold_metrics: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    best_params_list: list[dict | None] = []
    train_times: list[float] = []
    infer_times: list[float] = []
    trained_models: list[Any] = []

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        logger.info(
            "[%s] Fold %d/%d — train=%d, test=%d",
            model_name, fold_num, cv.n_splits, len(train_idx), len(test_idx),
        )

        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        w_tr = w_arr[train_idx]

        # ── Leakage Rule 1: scale fit on train only ─────────────────────
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # ── XGBoost: dynamic scale_pos_weight per fold ───────────────────
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        fold_ratio = n_neg / n_pos if n_pos > 0 else 1.0
        logger.debug(
            "[%s] Fold %d class ratio (neg/pos) = %.3f",
            model_name, fold_num, fold_ratio,
        )

        # Clone model so each fold starts fresh
        fold_model = clone(model)
        if model_name == "xgboost":
            fold_model.set_params(scale_pos_weight=fold_ratio)

        # ── Leakage Rule 4: nested hyperparameter tuning ─────────────────
        fold_best_params: dict | None = None
        if tune and model_name in config.ML_TUNING_SPACE:
            fold_best_params = _tune_model(
                fold_model, model_name,
                X_tr_sc, y_tr, t1.iloc[train_idx],
            )
            fold_model = clone(model)
            if model_name == "xgboost":
                fold_model.set_params(scale_pos_weight=fold_ratio)
            fold_model.set_params(**fold_best_params)
            logger.info(
                "[%s] Fold %d best params: %s", model_name, fold_num, fold_best_params,
            )

        best_params_list.append(fold_best_params)

        # ── Train ─────────────────────────────────────────────────────────
        t_start = time.perf_counter()
        fold_model.fit(X_tr_sc, y_tr, sample_weight=w_tr)
        train_time = time.perf_counter() - t_start
        train_times.append(train_time)

        # ── Leakage Rule 2: threshold optimised on TRAIN fold only ────────
        proba_train = fold_model.predict_proba(X_tr_sc)[:, 1]
        opt_threshold = _optimize_threshold(y_tr, proba_train)
        logger.debug(
            "[%s] Fold %d opt_threshold=%.4f", model_name, fold_num, opt_threshold,
        )

        # ── Inference on TEST fold ────────────────────────────────────────
        t_infer = time.perf_counter()
        proba_test = fold_model.predict_proba(X_te_sc)[:, 1]
        infer_time = time.perf_counter() - t_infer
        infer_times.append(infer_time)

        pred_test = (proba_test >= opt_threshold).astype(int)

        # ── Metrics on TEST fold ──────────────────────────────────────────
        metrics = _compute_metrics(y_te, pred_test, proba_test)
        metrics["threshold"] = opt_threshold
        metrics["fold"] = fold_num
        fold_metrics.append(metrics)

        logger.info(
            "[%s] Fold %d | ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  "
            "Prec=%.4f  Rec=%.4f  MCC=%.4f  train=%.1fs",
            model_name, fold_num,
            metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
            metrics["precision"], metrics["recall"], metrics["mcc"],
            train_time,
        )

        # Collect test-fold predictions with original DatetimeIndex
        test_index = X.index[test_idx]
        fold_preds = pd.DataFrame({
            "proba": proba_test,
            "pred": pred_test,
            "label": y_te,
            "fold": fold_num,
            "threshold": opt_threshold,
        }, index=test_index)
        all_predictions.append(fold_preds)

        trained_models.append(fold_model)
        del X_tr, X_te, X_tr_sc, X_te_sc, proba_train, proba_test
        gc.collect()

    # ── Aggregate across folds ────────────────────────────────────────────
    metric_keys = list(config.EVAL_METRICS)
    mean_metrics = {
        k: float(np.mean([fm[k] for fm in fold_metrics])) for k in metric_keys
    }
    std_metrics = {
        k: float(np.std([fm[k] for fm in fold_metrics], ddof=1)) for k in metric_keys
    }
    mean_metrics["mean_train_time_s"] = float(np.mean(train_times))
    mean_metrics["mean_infer_time_s"] = float(np.mean(infer_times))
    mean_metrics["total_train_time_s"] = float(np.sum(train_times))

    logger.info(
        "[%s] MEAN | ROC-AUC=%.4f±%.4f  PR-AUC=%.4f±%.4f  "
        "F1=%.4f±%.4f  MCC=%.4f±%.4f",
        model_name,
        mean_metrics["roc_auc"], std_metrics["roc_auc"],
        mean_metrics["pr_auc"],  std_metrics["pr_auc"],
        mean_metrics["f1"],      std_metrics["f1"],
        mean_metrics["mcc"],     std_metrics["mcc"],
    )

    predictions_df = pd.concat(all_predictions).sort_index()

    return {
        "fold_metrics":    fold_metrics,
        "mean_metrics":    mean_metrics,
        "std_metrics":     std_metrics,
        "predictions":     predictions_df,
        "best_params":     best_params_list,
        "train_times":     train_times,
        "infer_times":     infer_times,
        "model_name":      model_name,
        "trained_models":  trained_models,
    }


def _tune_model(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    t1_train: pd.Series,
) -> dict[str, Any]:
    """Run RandomizedSearchCV with inner PurgedEmbargoKFold.

    Tuning is performed entirely within the outer training fold so no
    test-fold information influences hyperparameter selection.

    Note: sample_weight is not passed to inner CV search to avoid sklearn
    metadata-routing complexity across versions.  Class imbalance is handled
    by class_weight / scale_pos_weight set on the model itself.

    Parameters
    ----------
    model : estimator
        Cloned unfitted model.
    model_name : str
        Key into ``config.ML_TUNING_SPACE``.
    X_train : np.ndarray
        Scaled training features (float32).
    y_train : np.ndarray
        Training labels.
    t1_train : pd.Series
        Label end-times for training events (used to build inner CV).

    Returns
    -------
    dict
        Best hyperparameters found by random search.
    """
    # Build a temporary feature DataFrame with the t1 index for inner CV
    idx = t1_train.index
    X_tmp = pd.DataFrame(X_train, index=idx)

    inner_cv = PurgedEmbargoKFold(
        n_splits=config.CV_N_SPLITS,
        t1=t1_train,
        embargo_pct=config.CV_PCT_EMBARGO,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=config.ML_TUNING_SPACE[model_name],
        n_iter=config.TUNING_N_ITER,
        cv=inner_cv,
        scoring=config.TUNING_SCORING,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,          # use all available cores on HPC
        refit=False,        # we rebuild the model ourselves with best params
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_tmp, y_train)

    return dict(search.best_params_)


# ======================================================================
# ORCHESTRATOR — all 3 models
# ======================================================================

def run_all_ml_models(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    weights: pd.Series,
    tune: bool = True,
) -> pd.DataFrame:
    """Train and evaluate LightGBM, XGBoost, and Random Forest.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (18 features), DatetimeIndex sorted ascending.
    y : pd.Series
        Binary labels aligned with X.
    t1 : pd.Series
        Label end-times aligned with X.
    weights : pd.Series
        Sample uniqueness weights aligned with X.
    tune : bool
        If True, run nested RandomizedSearchCV for each model.

    Returns
    -------
    pd.DataFrame
        One row per model.  Columns: model name + all metric mean/std pairs +
        runtime.  Suitable for paper Table 3.

    Side effects
    ------------
    Saves each model's best fold (highest PR-AUC) as a pickle under
    ``config.PROCESSED_DATA_DIR / "models" / "<model_name>_best.pkl"``.
    Saves ``results/tables/table3_ml_part.csv``.
    """
    # Global class ratio for initial model creation
    n_pos_total = int((y == 1).sum())
    n_neg_total = int((y == 0).sum())
    global_class_ratio = n_neg_total / n_pos_total if n_pos_total > 0 else 1.0
    logger.info(
        "Dataset: n=%d, pos=%d, neg=%d, global class ratio=%.3f",
        len(y), n_pos_total, n_neg_total, global_class_ratio,
    )

    # Shared outer CV splitter
    outer_cv = PurgedEmbargoKFold(
        n_splits=config.CV_N_SPLITS,
        t1=t1,
        embargo_pct=config.CV_PCT_EMBARGO,
    )

    models = create_ml_models(global_class_ratio)
    all_results: dict[str, dict] = {}

    for name, model_instance in models.items():
        logger.info("=" * 60)
        logger.info("Training %s (tune=%s)", name, tune)
        logger.info("=" * 60)

        result = train_evaluate_ml(
            X=X, y=y, t1=t1, weights=weights,
            model_name=name,
            model=model_instance,
            cv=outer_cv,
            tune=tune,
        )
        all_results[name] = result

        # Save best fold model (by PR-AUC on test fold)
        _save_best_model(result, name)

        del result["trained_models"]   # free memory
        gc.collect()

    # Build summary DataFrame for Table 3
    table3 = _build_table3(all_results)

    out_path = config.TABLES_DIR / "table3_ml_part.csv"
    table3.to_csv(out_path)
    logger.info("Table 3 ML part saved -> %s", out_path)

    return table3


def _save_best_model(result: dict[str, Any], model_name: str) -> None:
    """Persist the trained model from the fold with highest PR-AUC.

    Parameters
    ----------
    result : dict
        Output of ``train_evaluate_ml``.
    model_name : str
        Used to build the output filename.
    """
    fold_pr_aucs = [fm["pr_auc"] for fm in result["fold_metrics"]]
    best_fold_idx = int(np.argmax(fold_pr_aucs))
    best_model = result["trained_models"][best_fold_idx]

    models_dir = config.PROCESSED_DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = models_dir / f"{model_name}_best.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(best_model, f)

    logger.info(
        "Saved best %s model (fold %d, PR-AUC=%.4f) -> %s",
        model_name, best_fold_idx + 1, fold_pr_aucs[best_fold_idx], pkl_path,
    )


def _build_table3(all_results: dict[str, dict]) -> pd.DataFrame:
    """Construct summary DataFrame from run_all_ml_models results.

    Parameters
    ----------
    all_results : dict
        Map of model_name -> result dict from ``train_evaluate_ml``.

    Returns
    -------
    pd.DataFrame
        Rows = models; columns = mean/std for each metric + runtime.
    """
    rows = []
    for name, result in all_results.items():
        m = result["mean_metrics"]
        s = result["std_metrics"]
        row: dict[str, Any] = {
            "model": config.MODEL_DISPLAY_NAMES.get(name, name),
        }
        for metric in config.EVAL_METRICS:
            row[f"{metric}_mean"] = round(m[metric], 4)
            row[f"{metric}_std"]  = round(s[metric], 4)
        row["mean_train_time_s"] = round(m["mean_train_time_s"], 2)
        row["total_train_time_s"] = round(m["total_train_time_s"], 2)
        row["mean_infer_time_s"] = round(m["mean_infer_time_s"], 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df


# ======================================================================
# ENTRY POINT — run from project root
# ======================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(config.PROJECT_ROOT))

    from src.utils import set_reproducibility

    set_reproducibility(config.RANDOM_SEED)
    logger.info("Phase 5: ML model training | LIGHTWEIGHT_MODE=%s", config.LIGHTWEIGHT_MODE)

    # ── Load feature-label dataset ──────────────────────────────────────
    data_path = config.PROCESSED_DATA_DIR / "X_y_adaptive.parquet"
    if not data_path.exists():
        logger.error("Missing %s — run Phase 3 (features.py) first.", data_path)
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info("Loaded X_y_adaptive | shape=%s", df.shape)

    feature_cols = config.FEATURE_NAMES
    X = df[feature_cols]
    y = df["bin"].astype(int)
    t1 = df["t1"]
    weights = df["weight"]

    # Make sure t1 is a Series of timestamps
    if not pd.api.types.is_datetime64_any_dtype(t1):
        t1 = pd.to_datetime(t1)

    # ── Run all models ───────────────────────────────────────────────────
    table3 = run_all_ml_models(X, y, t1, weights, tune=True)

    print("\n" + "=" * 70)
    print("TABLE 3 — ML Models (Purged-Embargo CV, Adaptive Labels)")
    print("=" * 70)
    print(table3.to_string())
    print("=" * 70)
    print(f"\nSaved: {config.TABLES_DIR / 'table3_ml_part.csv'}")
