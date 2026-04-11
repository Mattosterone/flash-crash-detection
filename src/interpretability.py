"""
interpretability.py — Module 10: SHAP + Integrated Gradients interpretability.
Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection

Implements two complementary explanation frameworks:
  - SHAP (TreeExplainer) for tree-based models (LightGBM, XGBoost, RF)
  - Integrated Gradients (captum) for recurrent DL models (RNN, LSTM, GRU)

Bootstrapped 95% CIs on mean |SHAP| values allow uncertainty-aware feature
importance reporting in Table 6.

Public API
----------
shap_analysis(model, X, n_samples, n_bootstrap) → dict
    SHAP values + bootstrapped feature importance with 95% CI.

integrated_gradients_analysis(model, X_sequences, baseline, n_steps) → dict
    IG attributions aggregated to per-feature and per-timestep importance.

compare_feature_rankings(ml_importance, dl_importance, top_k) → pd.DataFrame
    Merge ML and DL importance rankings into Table 6.

run_interpretability_analysis(X, y, t1)
    Orchestrator: loads saved models, runs SHAP + IG, saves all figures
    and Table 6.  Called from notebook/Phase 8.
"""

import gc
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import config
from src.models_dl import CrashDetector, SequenceDataset
from src.utils import setup_logging

logger = setup_logging(__name__)

# Suppress SHAP and captum verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=UserWarning, module="captum")


# ======================================================================
# SHAP — TreeExplainer (tree-based ML models)
# ======================================================================

def shap_analysis(
    model: Any,
    X: pd.DataFrame,
    n_samples: int = 2000,
    n_bootstrap: int = 100,
) -> dict:
    """Compute SHAP values for a tree-based model with bootstrapped CIs.

    Uses ``shap.TreeExplainer`` which is exact (not approximate) for
    tree ensembles.  A random subsample of ``n_samples`` events is drawn
    without replacement (or all events if fewer available) to keep
    computation tractable.

    Bootstrap procedure:
        For each of ``n_bootstrap`` iterations resample the subsample with
        replacement, compute mean |SHAP| per feature, then report the mean
        and 2.5th/97.5th percentiles across bootstrap runs as the 95% CI.

    Parameters
    ----------
    model : sklearn-compatible fitted tree classifier
        LightGBM, XGBoost, or RandomForest model from models_ml.py.
    X : pd.DataFrame
        Feature matrix (n_events × 18 features), unscaled or pre-scaled —
        must match the representation the model was trained on.
    n_samples : int
        Number of events to subsample before computing SHAP.
        Reduced to config.SHAP_BACKGROUND_SAMPLES in LIGHTWEIGHT_MODE via
        the calling orchestrator.
    n_bootstrap : int
        Number of bootstrap resamples for 95% CI computation.

    Returns
    -------
    dict with keys:
        ``shap_values``       : np.ndarray (n_samples, n_features) — raw SHAP
        ``feature_importance``: pd.DataFrame — columns: feature, mean_abs_shap,
                                ci_low, ci_high, rank
        ``X_sample``          : pd.DataFrame — the subsampled feature matrix
        ``explainer``         : shap.TreeExplainer (for downstream plot calls)
    """
    import shap

    feature_names = list(X.columns)
    n_events = len(X)

    if n_samples >= n_events:
        logger.info(
            "SHAP: n_samples=%d >= n_events=%d — using all events",
            n_samples, n_events,
        )
        X_sample = X.copy()
    else:
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = rng.choice(n_events, size=n_samples, replace=False)
        idx = np.sort(idx)   # preserve temporal order
        X_sample = X.iloc[idx].copy()

    logger.info(
        "SHAP: fitting TreeExplainer on %d events (%d features)",
        len(X_sample), len(feature_names),
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Normalise output shape to (n_samples, n_features):
    #   - list of two arrays [neg, pos]  → take index 1 (crash class)
    #   - 3D ndarray (n, f, 2)           → take [:, :, 1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    logger.info("SHAP values computed: shape=%s", shap_values.shape)

    # ── Global feature importance: mean |SHAP| with bootstrap CI ──────────
    abs_shap = np.abs(shap_values)   # (n_samples, n_features)
    mean_abs = abs_shap.mean(axis=0)  # (n_features,)

    rng = np.random.default_rng(config.RANDOM_SEED + 1)
    boot_means = np.empty((n_bootstrap, len(feature_names)))
    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, len(X_sample), size=len(X_sample))
        boot_means[b] = abs_shap[boot_idx].mean(axis=0)

    ci_low  = np.percentile(boot_means, 2.5,  axis=0)
    ci_high = np.percentile(boot_means, 97.5, axis=0)

    feature_importance = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
        "ci_low":        ci_low,
        "ci_high":       ci_high,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    feature_importance["rank"] = np.arange(1, len(feature_names) + 1)

    logger.info(
        "Top-5 SHAP features: %s",
        feature_importance["feature"].head(5).tolist(),
    )

    return {
        "shap_values":        shap_values,
        "feature_importance": feature_importance,
        "X_sample":           X_sample,
        "explainer":          explainer,
    }


# ======================================================================
# INTEGRATED GRADIENTS — captum (recurrent DL models)
# ======================================================================

def integrated_gradients_analysis(
    model: "CrashDetector",
    X_sequences: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = config.IG_N_STEPS,
) -> dict:
    """Compute Integrated Gradients attributions for a recurrent model.

    Attributions are computed for each sequence in ``X_sequences`` using
    the Riemann approximation with ``n_steps`` interpolation points.  The
    baseline is the zero tensor (no-signal reference) by default, or the
    mean sequence if ``config.IG_BASELINE == "mean"``.

    Aggregation:
        - ``feature_importance``: mean |attribution| averaged over all
          sequences AND over the time dimension → one score per feature.
        - ``temporal_importance``: mean |attribution| averaged over
          sequences AND features → one score per timestep in the lookback.

    Parameters
    ----------
    model : CrashDetector
        Trained recurrent model in eval mode.
    X_sequences : np.ndarray
        Shape (n_sequences, lookback, n_features) — already scaled.
    baseline : np.ndarray, optional
        Reference input tensor, shape matching X_sequences[0].
        If None uses zero baseline (or mean if config.IG_BASELINE=="mean").
    n_steps : int
        Number of Riemann approximation steps (default: config.IG_N_STEPS).

    Returns
    -------
    dict with keys:
        ``attributions``      : np.ndarray (n_sequences, lookback, n_features)
        ``feature_importance``: pd.DataFrame — columns: feature, mean_abs_ig, rank
        ``temporal_importance``: pd.DataFrame — columns: timestep, mean_abs_ig
    """
    from captum.attr import IntegratedGradients

    feature_names = config.FEATURE_NAMES
    n_seq, lookback, n_feat = X_sequences.shape

    logger.info(
        "IG: computing attributions for %d sequences (lookback=%d, n_feat=%d, steps=%d)",
        n_seq, lookback, n_feat, n_steps,
    )

    # Determine device from model parameters
    device = next(model.parameters()).device

    model.eval()

    # ── Baseline ──────────────────────────────────────────────────────────
    if baseline is None:
        if config.IG_BASELINE == "mean":
            baseline_arr = X_sequences.mean(axis=0, keepdims=True)  # (1, L, F)
            baseline_tensor = torch.tensor(
                np.repeat(baseline_arr, n_seq, axis=0),
                dtype=torch.float32,
            ).to(device)
        else:
            # Zero baseline
            baseline_tensor = torch.zeros(
                (n_seq, lookback, n_feat), dtype=torch.float32
            ).to(device)
    else:
        baseline_tensor = torch.tensor(
            np.broadcast_to(baseline, (n_seq, lookback, n_feat)).copy(),
            dtype=torch.float32,
        ).to(device)

    input_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(device)

    # captum wrapper: model must output a scalar per sample
    # CrashDetector outputs (batch, 1) — squeeze to (batch,)
    def _model_wrapper(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pass  # no-op context, IG handles grad internally
        return model(x).squeeze(1)   # (batch,)

    ig = IntegratedGradients(_model_wrapper)

    # Process in mini-batches to avoid OOM on long sequence arrays
    batch_size = 64
    all_attrs = []

    for start in range(0, n_seq, batch_size):
        end = min(start + batch_size, n_seq)
        inp_b  = input_tensor[start:end].requires_grad_(True)
        base_b = baseline_tensor[start:end]

        attrs, _ = ig.attribute(
            inputs=inp_b,
            baselines=base_b,
            n_steps=n_steps,
            return_convergence_delta=True,
            internal_batch_size=batch_size,
        )
        all_attrs.append(attrs.detach().cpu().numpy())

        del inp_b, base_b, attrs
        gc.collect()

    attributions = np.concatenate(all_attrs, axis=0)   # (n_seq, lookback, n_feat)
    logger.info("IG attributions computed: shape=%s", attributions.shape)

    # ── Feature importance: mean |attribution| over sequences and time ─────
    abs_attr = np.abs(attributions)  # (n_seq, lookback, n_feat)
    feat_mean_abs = abs_attr.mean(axis=(0, 1))   # (n_feat,)

    feature_importance = pd.DataFrame({
        "feature":     feature_names,
        "mean_abs_ig": feat_mean_abs,
    }).sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)
    feature_importance["rank"] = np.arange(1, len(feature_names) + 1)

    # ── Temporal importance: mean |attribution| over sequences and features ─
    temporal_mean_abs = abs_attr.mean(axis=(0, 2))   # (lookback,)
    temporal_importance = pd.DataFrame({
        "timestep":    np.arange(-lookback + 1, 1),   # -L+1, ..., 0 (0 = most recent)
        "mean_abs_ig": temporal_mean_abs,
    })

    logger.info(
        "Top-5 IG features: %s",
        feature_importance["feature"].head(5).tolist(),
    )

    return {
        "attributions":        attributions,
        "feature_importance":  feature_importance,
        "temporal_importance": temporal_importance,
    }


# ======================================================================
# TABLE 6 — feature ranking comparison
# ======================================================================

def compare_feature_rankings(
    ml_importance: pd.DataFrame,
    dl_importance: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Build Table 6: top-k feature rankings from ML (SHAP) and DL (IG).

    Parameters
    ----------
    ml_importance : pd.DataFrame
        Output of shap_analysis()["feature_importance"].
        Must have columns: ``feature``, ``mean_abs_shap``, ``rank``.
    dl_importance : pd.DataFrame
        Output of integrated_gradients_analysis()["feature_importance"].
        Must have columns: ``feature``, ``mean_abs_ig``, ``rank``.
    top_k : int
        Number of top features to include from each model family.

    Returns
    -------
    pd.DataFrame
        Columns: feature, ml_rank, ml_mean_abs_shap, dl_rank, dl_mean_abs_ig.
        Union of top-k features from both methods, sorted by ML rank.
    """
    ml_top = ml_importance.head(top_k)[["feature", "mean_abs_shap", "rank"]].copy()
    ml_top = ml_top.rename(columns={"rank": "ml_rank", "mean_abs_shap": "ml_mean_abs_shap"})

    dl_top = dl_importance.head(top_k)[["feature", "mean_abs_ig", "rank"]].copy()
    dl_top = dl_top.rename(columns={"rank": "dl_rank", "mean_abs_ig": "dl_mean_abs_ig"})

    # Outer merge: union of top-k from both
    merged = ml_top.merge(dl_top, on="feature", how="outer")

    # Fill ranks for features only in one set
    ml_all_ranks = ml_importance.set_index("feature")["rank"]
    dl_all_ranks = dl_importance.set_index("feature")["rank"]

    for i, row in merged.iterrows():
        feat = row["feature"]
        if pd.isna(row.get("ml_rank")):
            merged.at[i, "ml_rank"] = int(ml_all_ranks.get(feat, len(ml_importance) + 1))
            merged.at[i, "ml_mean_abs_shap"] = float(
                ml_importance.loc[ml_importance["feature"] == feat, "mean_abs_shap"].iloc[0]
                if feat in ml_importance["feature"].values else np.nan
            )
        if pd.isna(row.get("dl_rank")):
            merged.at[i, "dl_rank"] = int(dl_all_ranks.get(feat, len(dl_importance) + 1))
            merged.at[i, "dl_mean_abs_ig"] = float(
                dl_importance.loc[dl_importance["feature"] == feat, "mean_abs_ig"].iloc[0]
                if feat in dl_importance["feature"].values else np.nan
            )

    merged["ml_rank"] = merged["ml_rank"].astype(int)
    merged["dl_rank"] = merged["dl_rank"].astype(int)
    merged = merged.sort_values("ml_rank").reset_index(drop=True)

    # Round importance values for readability
    merged["ml_mean_abs_shap"] = merged["ml_mean_abs_shap"].round(6)
    merged["dl_mean_abs_ig"]   = merged["dl_mean_abs_ig"].round(6)

    logger.info("Table 6 built: %d features in union of top-%d", len(merged), top_k)
    return merged


# ======================================================================
# FIGURE 7 — SHAP beeswarm summary (best tree-based model)
# ======================================================================

def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Figure 7: SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape (n_samples, n_features) — positive-class SHAP values.
    X_sample : pd.DataFrame
        Subsampled feature matrix aligned with shap_values.
    model_name : str
        Used in the figure title (e.g. "LightGBM").
    save_path : Path, optional
        Output path.  Defaults to config.FIGURES_DIR / "figure7_shap_summary.pdf".
    """
    import shap

    if save_path is None:
        save_path = config.FIGURES_DIR / "figure7_shap_summary.pdf"

    plt.style.use(config.FIGURE_STYLE)
    plt.rcParams.update({
        "font.size":        config.FIGURE_FONT_SIZE,
        "axes.titlesize":   config.FIGURE_TITLE_SIZE,
        "axes.labelsize":   config.FIGURE_LABEL_SIZE,
        "xtick.labelsize":  config.FIGURE_TICK_SIZE,
        "ytick.labelsize":  config.FIGURE_TICK_SIZE,
        "legend.fontsize":  config.FIGURE_LEGEND_SIZE,
    })

    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=config.SHAP_MAX_DISPLAY,
        show=False,
        plot_type="dot",
    )
    plt.title(
        f"SHAP Feature Importance — {model_name}\n"
        f"(n={len(X_sample)} events, positive class = Flash Crash)",
        fontsize=config.FIGURE_TITLE_SIZE,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info("Figure 7 (SHAP beeswarm) saved → %s", save_path)


# ======================================================================
# FIGURE 8 — IG attribution summary (best recurrent model)
# ======================================================================

def plot_ig_summary(
    ig_result: dict,
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Figure 8: Integrated Gradients attribution summary.

    Two-panel figure:
        Left  — horizontal bar chart of mean |IG| per feature (all 18).
        Right — temporal importance: mean |IG| vs lookback timestep.

    Parameters
    ----------
    ig_result : dict
        Output of integrated_gradients_analysis().
    model_name : str
        Used in the figure title (e.g. "LSTM").
    save_path : Path, optional
        Output path.  Defaults to config.FIGURES_DIR / "figure8_ig_summary.pdf".
    """
    if save_path is None:
        save_path = config.FIGURES_DIR / "figure8_ig_summary.pdf"

    feat_imp  = ig_result["feature_importance"]
    temp_imp  = ig_result["temporal_importance"]

    plt.style.use(config.FIGURE_STYLE)
    plt.rcParams.update({
        "font.size":        config.FIGURE_FONT_SIZE,
        "axes.titlesize":   config.FIGURE_TITLE_SIZE,
        "axes.labelsize":   config.FIGURE_LABEL_SIZE,
        "xtick.labelsize":  config.FIGURE_TICK_SIZE,
        "ytick.labelsize":  config.FIGURE_TICK_SIZE,
        "legend.fontsize":  config.FIGURE_LEGEND_SIZE,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # ── Left: per-feature bar chart ─────────────────────────────────────
    feat_sorted = feat_imp.sort_values("mean_abs_ig", ascending=True)
    ax1.barh(
        feat_sorted["feature"],
        feat_sorted["mean_abs_ig"],
        color="#56B4E9",
        edgecolor="none",
    )
    ax1.set_xlabel("Mean |Attribution|")
    ax1.set_title(f"Feature Importance\n({model_name}, Integrated Gradients)")
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # ── Right: temporal importance ───────────────────────────────────────
    ax2.plot(
        temp_imp["timestep"],
        temp_imp["mean_abs_ig"],
        color="#0072B2",
        linewidth=1.8,
        marker="o",
        markersize=4,
    )
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8, label="Event time")
    ax2.set_xlabel("Timestep relative to event (bars)")
    ax2.set_ylabel("Mean |Attribution|")
    ax2.set_title(f"Temporal Attribution Pattern\n({model_name})")
    ax2.legend()
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.suptitle(
        f"Integrated Gradients Interpretability — {model_name}",
        fontsize=config.FIGURE_TITLE_SIZE + 1,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info("Figure 8 (IG summary) saved → %s", save_path)


# ======================================================================
# FIGURE 9 — Local explanation for a representative crash event
# ======================================================================

def plot_local_explanation(
    shap_result: dict,
    event_idx: Optional[int] = None,
    model_name: str = "LightGBM",
    save_path: Optional[Path] = None,
) -> None:
    """Figure 9: Local SHAP waterfall plot for a single crash event.

    Selects a representative crash event (the event where the model's
    crash probability is highest, i.e., the most "prototypical" crash)
    unless ``event_idx`` is supplied explicitly.

    Parameters
    ----------
    shap_result : dict
        Output of shap_analysis().
    event_idx : int, optional
        Row index into X_sample.  If None, picks the event with the
        largest sum of positive SHAP values (most crash-like).
    model_name : str
        Used in the figure title.
    save_path : Path, optional
        Output path.  Defaults to config.FIGURES_DIR / "figure9_local_explanation.pdf".
    """
    import shap

    if save_path is None:
        save_path = config.FIGURES_DIR / "figure9_local_explanation.pdf"

    shap_values = shap_result["shap_values"]     # (n_samples, n_features)
    X_sample    = shap_result["X_sample"]        # pd.DataFrame
    explainer   = shap_result["explainer"]

    # Select most prototypical crash: largest sum of positive SHAP contributions
    if event_idx is None:
        pos_contrib = np.maximum(shap_values, 0).sum(axis=1)  # (n_samples,)
        event_idx = int(np.argmax(pos_contrib))
        logger.info(
            "Local explanation: selected event_idx=%d "
            "(largest positive SHAP contribution sum)",
            event_idx,
        )

    plt.style.use(config.FIGURE_STYLE)
    plt.rcParams.update({
        "font.size":        config.FIGURE_FONT_SIZE,
        "axes.titlesize":   config.FIGURE_TITLE_SIZE,
        "axes.labelsize":   config.FIGURE_LABEL_SIZE,
        "xtick.labelsize":  config.FIGURE_TICK_SIZE,
        "ytick.labelsize":  config.FIGURE_TICK_SIZE,
    })

    # Build a shap.Explanation object for waterfall
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        base_value = float(ev[1])   # positive (crash) class
    else:
        base_value = float(ev)
    explanation = shap.Explanation(
        values=shap_values[event_idx],
        base_values=base_value,
        data=X_sample.iloc[event_idx].values,
        feature_names=list(X_sample.columns),
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)

    ts_label = ""
    if hasattr(X_sample.index, "strftime"):
        ts_label = f" — {X_sample.index[event_idx].strftime('%Y-%m-%d %H:%M')}"
    plt.title(
        f"Local SHAP Explanation — {model_name}{ts_label}\n"
        f"(Representative crash event, event index={event_idx})",
        fontsize=config.FIGURE_TITLE_SIZE,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info("Figure 9 (local explanation) saved → %s", save_path)


# ======================================================================
# MODEL LOADING HELPERS
# ======================================================================

def _load_best_ml_model(model_name: str) -> Any:
    """Load the best ML model from the saved pickle.

    Parameters
    ----------
    model_name : str
        One of ``"lgbm"``, ``"xgboost"``, ``"rf"``.

    Returns
    -------
    Fitted sklearn-compatible classifier.

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist (Phase 5 not yet run).
    """
    pkl_path = config.PROCESSED_DATA_DIR / "models" / f"{model_name}_best.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Saved ML model not found: {pkl_path}\n"
            f"Run Phase 5 (models_ml.py) to generate it."
        )
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded best %s model ← %s", model_name, pkl_path)
    return model


def _load_best_dl_model(model_name: str) -> "CrashDetector":
    """Reconstruct the best DL model from its saved state_dict.

    Parameters
    ----------
    model_name : str
        One of ``"rnn"``, ``"lstm"``, ``"gru"``.

    Returns
    -------
    CrashDetector
        Model in eval mode, on CPU (interpretability does not require GPU).

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist (Phase 6 not yet run).
    """
    pkl_path = config.PROCESSED_DATA_DIR / "models" / f"{model_name}_best.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Saved DL model not found: {pkl_path}\n"
            f"Run Phase 6 (models_dl.py) to generate it."
        )
    with open(pkl_path, "rb") as f:
        state_dict = pickle.load(f)

    rnn_type = model_name.upper()   # "rnn" → "RNN", etc.
    model = CrashDetector(
        input_size=config.N_FEATURES,
        hidden_size=config.DL_HIDDEN_SIZE,
        num_layers=config.DL_NUM_LAYERS,
        dropout=config.DL_DROPOUT,
        rnn_type=rnn_type,
    )
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded best %s model ← %s", model_name, pkl_path)
    return model


# ======================================================================
# SEQUENCE CONSTRUCTION FOR IG
# ======================================================================

def _build_sequences_for_ig(
    X_scaled: np.ndarray,
    n_samples: int,
    lookback: int,
) -> np.ndarray:
    """Sample n_samples fixed-length sequences from a scaled feature array.

    Sequences are sampled from valid endpoints (index >= lookback - 1).
    Temporal order is respected; endpoints are sampled uniformly at random
    from all valid positions in the array.

    Parameters
    ----------
    X_scaled : np.ndarray
        Shape (n_events, n_features) — already StandardScaler-transformed.
    n_samples : int
        Number of sequences to extract.
    lookback : int
        Sequence length in bars.

    Returns
    -------
    np.ndarray
        Shape (n_samples, lookback, n_features).
    """
    n_events = len(X_scaled)
    valid_endpoints = np.arange(lookback - 1, n_events)

    if n_samples >= len(valid_endpoints):
        sampled = valid_endpoints
    else:
        rng = np.random.default_rng(config.RANDOM_SEED + 2)
        sampled = rng.choice(valid_endpoints, size=n_samples, replace=False)
        sampled = np.sort(sampled)

    seqs = np.stack(
        [X_scaled[e - lookback + 1 : e + 1] for e in sampled],
        axis=0,
    )   # (n_samples, lookback, n_features)
    logger.info(
        "Built %d sequences (lookback=%d, n_feat=%d) for IG",
        len(seqs), lookback, X_scaled.shape[1],
    )
    return seqs


# ======================================================================
# ORCHESTRATOR — run full Phase 8 interpretability pipeline
# ======================================================================

def run_interpretability_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
) -> None:
    """Full Phase 8 orchestrator: load models, run SHAP + IG, save outputs.

    Leakage note: SHAP and IG are computed on TRAINING data from the best
    fold (the fold used to save the model).  The interpretation reflects
    what the model learned, not held-out test performance.  No future
    labels or test-fold features are used.

    Steps:
        1. Scale X using a StandardScaler fit on a representative training
           portion (last N events, excluding the final fold).
        2. Load best ML model (highest PR-AUC across all ML models).
        3. Run SHAP analysis → Figure 7 + local explanation → Figure 9.
        4. Load best DL model (highest PR-AUC across all DL models).
        5. Run IG analysis → Figure 8.
        6. Compare rankings → Table 6.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (all events, 18 features), DatetimeIndex.
    y : pd.Series
        Binary labels aligned with X.
    t1 : pd.Series
        Label end-times (used only for leakage-check logging here).
    """
    # ── Step 1: determine best ML and DL models ───────────────────────────
    ml_model_names = ["lgbm", "xgboost", "rf"]
    dl_model_names = ["rnn", "lstm", "gru"]

    best_ml_name = _pick_best_model(ml_model_names, table_stem="table3_ml_part")
    best_dl_name = _pick_best_model(dl_model_names, table_stem="table3_dl_part")

    logger.info(
        "Best ML model: %s | Best DL model: %s",
        best_ml_name, best_dl_name,
    )

    # ── Step 2: fit scaler on training portion (first 80% by time) ───────
    n_train = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_arr = X.values.astype(np.float32)
    scaler.fit(X_arr[:n_train])
    X_scaled_arr = scaler.transform(X_arr).astype(np.float32)
    X_scaled_df = pd.DataFrame(X_scaled_arr, index=X.index, columns=X.columns)

    # SHAP sample size
    shap_n = (
        config.SHAP_BACKGROUND_SAMPLES * 5   # 500 in lightweight → use background * 5
        if config.LIGHTWEIGHT_MODE
        else 2000
    )
    shap_n = min(shap_n, len(X_scaled_df))

    # ── Step 3: SHAP on best ML model ─────────────────────────────────────
    logger.info("=== SHAP Analysis (%s) ===", best_ml_name)
    ml_model = _load_best_ml_model(best_ml_name)
    shap_result = shap_analysis(
        model=ml_model,
        X=X_scaled_df,
        n_samples=shap_n,
        n_bootstrap=100,
    )

    # Figure 7: SHAP beeswarm
    plot_shap_summary(
        shap_values=shap_result["shap_values"],
        X_sample=shap_result["X_sample"],
        model_name=config.MODEL_DISPLAY_NAMES.get(best_ml_name, best_ml_name),
    )

    # Figure 9: local explanation for most prototypical crash
    plot_local_explanation(
        shap_result=shap_result,
        model_name=config.MODEL_DISPLAY_NAMES.get(best_ml_name, best_ml_name),
    )

    del ml_model
    gc.collect()

    # ── Step 4: IG on best DL model ───────────────────────────────────────
    logger.info("=== Integrated Gradients Analysis (%s) ===", best_dl_name)
    dl_model = _load_best_dl_model(best_dl_name)

    lookback = config.DL_LOOKBACK
    ig_n = min(shap_n, len(X_scaled_arr) - lookback)
    X_seqs = _build_sequences_for_ig(X_scaled_arr, n_samples=ig_n, lookback=lookback)

    ig_result = integrated_gradients_analysis(
        model=dl_model,
        X_sequences=X_seqs,
        n_steps=config.IG_N_STEPS,
    )

    # Figure 8: IG summary
    plot_ig_summary(
        ig_result=ig_result,
        model_name=config.MODEL_DISPLAY_NAMES.get(best_dl_name, best_dl_name),
    )

    del dl_model, X_seqs
    gc.collect()

    # ── Step 5: Table 6 — feature ranking comparison ─────────────────────
    logger.info("=== Building Table 6 (feature ranking comparison) ===")
    table6 = compare_feature_rankings(
        ml_importance=shap_result["feature_importance"],
        dl_importance=ig_result["feature_importance"],
        top_k=5,
    )
    table6_path = config.TABLES_DIR / "table6_feature_rankings.csv"
    table6.to_csv(table6_path, index=False)
    logger.info("Table 6 saved → %s", table6_path)
    logger.info("\n%s", table6.to_string())

    # ── Step 6: Save full importance tables ───────────────────────────────
    shap_imp_path = config.TABLES_DIR / "shap_feature_importance.csv"
    shap_result["feature_importance"].to_csv(shap_imp_path, index=False)
    logger.info("SHAP feature importance table saved → %s", shap_imp_path)

    ig_imp_path = config.TABLES_DIR / "ig_feature_importance.csv"
    ig_result["feature_importance"].to_csv(ig_imp_path, index=False)
    logger.info("IG feature importance table saved → %s", ig_imp_path)

    logger.info("=== Phase 8 interpretability analysis complete ===")


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _pick_best_model(model_names: list[str], table_stem: str) -> str:
    """Select the model with the highest mean PR-AUC from a saved table.

    Falls back to the first model name if the table is missing.

    Parameters
    ----------
    model_names : list[str]
        Candidate model names (e.g. ``["lgbm", "xgboost", "rf"]``).
    table_stem : str
        CSV file stem in config.TABLES_DIR (without ``.csv``).

    Returns
    -------
    str
        Model name key (lowercase, e.g. ``"lgbm"``).
    """
    table_path = config.TABLES_DIR / f"{table_stem}.csv"
    if not table_path.exists():
        logger.warning(
            "Table not found: %s — defaulting to %s",
            table_path, model_names[0],
        )
        return model_names[0]

    df = pd.read_csv(table_path, index_col=0)

    # Table index uses display names; build reverse map
    reverse_names = {v: k for k, v in config.MODEL_DISPLAY_NAMES.items()}

    best_name = model_names[0]
    best_pr_auc = -1.0
    pr_col = "pr_auc_mean"
    if pr_col not in df.columns:
        logger.warning(
            "Column '%s' not found in %s — defaulting to %s",
            pr_col, table_path, model_names[0],
        )
        return model_names[0]

    for display_name, pr_auc in df[pr_col].items():
        key = reverse_names.get(str(display_name), str(display_name).lower())
        if key in model_names and pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = key

    logger.info(
        "Best model from %s: %s (pr_auc_mean=%.4f)",
        table_path.name, best_name, best_pr_auc,
    )
    return best_name


# ======================================================================
# ENTRY POINT — run from project root
# ======================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(config.PROJECT_ROOT))

    from src.utils import set_reproducibility, load_result

    set_reproducibility(config.RANDOM_SEED)
    logger.info(
        "Phase 8: Interpretability | LIGHTWEIGHT_MODE=%s",
        config.LIGHTWEIGHT_MODE,
    )

    # Load feature-label dataset (adaptive labels, same as Phase 5/6)
    data_path = config.PROCESSED_DATA_DIR / "X_y_adaptive.parquet"
    if not data_path.exists():
        logger.error(
            "Missing %s — run Phase 3 (features.py) first.", data_path
        )
        sys.exit(1)

    logger.info("Loading X_y_adaptive.parquet …")
    data = load_result("X_y_adaptive")
    feature_cols = config.FEATURE_NAMES
    X_full = data[feature_cols]
    y_full = data["bin"]
    t1_full = data["t1"]

    del data
    gc.collect()

    run_interpretability_analysis(X_full, y_full, t1_full)
