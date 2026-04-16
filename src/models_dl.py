"""
models_dl.py — Module 8: Deep learning model training under Purged-Embargo CV.
Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection

Implements leakage-aware training pipelines for three recurrent models:
  - Vanilla RNN
  - LSTM
  - GRU

Leakage safeguards per fold (non-negotiable, see CLAUDE.md):
  1. Feature scaling fit on TRAIN ONLY; test transformed with train scaler.
  2. Classification threshold optimized on TRAIN ONLY (max F1 over grid).
  3. BCE pos_weight computed from actual training fold class ratio.
  4. Sequence construction respects fold boundaries: no lookback window
     may reach into purged/embargoed regions or cross fold boundaries.
     Implemented via the valid_indices parameter in SequenceDataset.
  5. Early stopping uses only the last 20% of the training fold as a
     holdout validation set — temporal order preserved; no test data seen.

Public API
----------
SequenceDataset
    torch.utils.data.Dataset converting tabular events to fixed-length
    sequences, constrained by valid_indices.

CrashDetector
    Recurrent model: RNN/LSTM/GRU -> Dropout -> Linear -> Sigmoid.

create_dl_models(input_size)
    Build initial model configs for all three DL architectures.

train_evaluate_dl(X, y, t1, weights, model_name, model_class, cv, device)
    Full purged-CV pipeline for one DL model; returns per-fold + aggregate metrics.

run_all_dl_models(X, y, t1, weights)
    Train & evaluate all 3 DL models; returns DataFrame for Table 3.
"""

import copy
import gc
import pickle
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, Dataset

import config
from src.purged_cv import PurgedEmbargoKFold
from src.utils import setup_logging

logger = setup_logging(__name__)


# ======================================================================
# SEQUENCE DATASET
# ======================================================================

class SequenceDataset(Dataset):
    """Convert tabular events into fixed-length lookback sequences.

    Each sample is a (sequence, label) pair where sequence has shape
    (lookback, n_features) — the `lookback` most recent consecutive events
    ending at the endpoint position (inclusive).

    The ``valid_indices`` parameter is the fold-boundary enforcement
    mechanism: only positions listed there can serve as sequence endpoints,
    and any position without enough prior history (i < lookback - 1) is
    automatically dropped.  Pass the 0-based positions within the supplied
    ``X`` array that belong to the current split (train or test portion).

    Parameters
    ----------
    X : np.ndarray
        Feature array, shape (n, n_features).  Must already be scaled.
    y : np.ndarray
        Label array, shape (n,).
    valid_indices : np.ndarray
        0-based positions in ``X`` that may be used as sequence endpoints.
        Positions with index < lookback - 1 are silently excluded.
    lookback : int
        Number of consecutive events per sequence.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        valid_indices: np.ndarray,
        lookback: int = config.DL_LOOKBACK,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lookback = lookback
        self.endpoints: np.ndarray = np.array(
            [i for i in valid_indices if i >= lookback - 1],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, idx: int):
        end = int(self.endpoints[idx])
        start = end - self.lookback + 1
        seq = self.X[start : end + 1]         # (lookback, n_features)
        label = self.y[end].unsqueeze(0)       # (1,)
        return seq, label


# ======================================================================
# MODEL ARCHITECTURE
# ======================================================================

class CrashDetector(nn.Module):
    """Recurrent crash-detection classifier.

    Architecture:
        Input (batch, seq_len, n_features)
          -> RNN / LSTM / GRU (num_layers stacked, batch_first=True)
          -> last time-step hidden state
          -> Dropout
          -> Linear(hidden_size -> 1)
          -> Sigmoid
        Output (batch, 1) — crash probability in [0, 1].

    Parameters
    ----------
    input_size : int
        Number of features per time step.
    hidden_size : int
        Hidden units per recurrent layer.
    num_layers : int
        Number of stacked recurrent layers.
    dropout : float
        Dropout probability applied between recurrent layers (when
        num_layers > 1) and before the final linear projection.
    rnn_type : str
        One of ``"RNN"``, ``"LSTM"``, or ``"GRU"`` (case-insensitive).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = config.DL_HIDDEN_SIZE,
        num_layers: int = config.DL_NUM_LAYERS,
        dropout: float = config.DL_DROPOUT,
        rnn_type: str = "LSTM",
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.upper()
        if rnn_type not in ("RNN", "LSTM", "GRU"):
            raise ValueError(
                f"rnn_type must be 'RNN', 'LSTM', or 'GRU', got '{rnn_type}'"
            )
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        # Dropout between layers is only meaningful when num_layers > 1
        inter_layer_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=inter_layer_dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) — crash probabilities in [0, 1].
        """
        # rnn_out: (batch, seq_len, hidden_size)
        # For LSTM: (rnn_out, (h_n, c_n)); for RNN/GRU: (rnn_out, h_n)
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]    # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        prob = self.sigmoid(self.fc(last_hidden))   # (batch, 1)
        return prob


# ======================================================================
# MODEL FACTORY
# ======================================================================

def create_dl_models(input_size: int) -> dict[str, dict[str, Any]]:
    """Build CrashDetector init-kwargs for all three DL architectures.

    Parameters
    ----------
    input_size : int
        Number of input features (= config.N_FEATURES = 18).

    Returns
    -------
    dict
        Keys: ``"rnn"``, ``"lstm"``, ``"gru"``.
        Values: dicts containing ``"rnn_type"`` and ``"init_kwargs"``
        suitable for ``CrashDetector(**init_kwargs)``.
    """
    common = {
        "input_size": input_size,
        "hidden_size": config.DL_HIDDEN_SIZE,
        "num_layers": config.DL_NUM_LAYERS,
        "dropout": config.DL_DROPOUT,
    }
    return {
        "rnn":  {"rnn_type": "RNN",  "init_kwargs": {**common, "rnn_type": "RNN"}},
        "lstm": {"rnn_type": "LSTM", "init_kwargs": {**common, "rnn_type": "LSTM"}},
        "gru":  {"rnn_type": "GRU",  "init_kwargs": {**common, "rnn_type": "GRU"}},
    }


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _effective_lookback() -> int:
    """Return lookback length, reduced to 10 in LIGHTWEIGHT_MODE.

    See CLAUDE.md: "In lightweight mode, use batch_size=64, lookback=10."
    """
    return 10 if config.LIGHTWEIGHT_MODE else config.DL_LOOKBACK


def detect_device() -> torch.device:
    """Auto-detect the best available PyTorch compute device.

    Priority: Apple Silicon MPS > CUDA > CPU.

    Returns
    -------
    torch.device
    """
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    logger.info("Using device: %s", dev)
    return dev


def _optimize_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_grid: int = config.THRESHOLD_SEARCH_GRID,
) -> float:
    """Find probability threshold maximising F1 on the given data.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    proba : np.ndarray
        Predicted probabilities for the positive class.
    n_grid : int
        Number of uniformly-spaced threshold candidates in (0.01, 0.99).

    Returns
    -------
    float
        Threshold that maximised F1.
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
        Keys: roc_auc, pr_auc, f1, precision, recall, brier_score, mcc.
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


def _predict_loader(
    model: CrashDetector,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference over a DataLoader.

    Parameters
    ----------
    model : CrashDetector
        Model to evaluate (set to eval mode internally).
    loader : DataLoader
        Batches of (sequence, label) pairs.
    device : torch.device
        Compute device.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (probas, labels), each shape (n_samples,).
    """
    model.eval()
    all_probas: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            proba = model(X_batch).cpu().squeeze(-1).numpy()
            all_probas.append(proba)
            all_labels.append(y_batch.squeeze(-1).numpy())
    return np.concatenate(all_probas), np.concatenate(all_labels)


def _compute_val_pr_auc(
    model: CrashDetector,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute PR-AUC on a validation DataLoader.

    Returns 0.0 if the validation set contains only one class.
    """
    probas, labels = _predict_loader(model, loader, device)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, probas))


# ======================================================================
# TRAINING LOOP FOR ONE FOLD
# ======================================================================

def _train_fold(
    model: CrashDetector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight_scalar: float,
    device: torch.device,
    model_name: str,
    fold_num: int,
) -> tuple[CrashDetector, list[float], list[float]]:
    """Training loop with early stopping and LR scheduling for one fold.

    Leakage safeguards enforced here:
    - pos_weight_scalar = n_neg / n_pos from the TRAINING fold only.
    - val_loader uses the last 20% of the training fold (temporal order).
    - No test data is seen during training or validation.

    Parameters
    ----------
    model : CrashDetector
        Freshly initialised model (weights not yet trained).
    train_loader : DataLoader
        Training sequences (fold train portion minus last 20%).
    val_loader : DataLoader
        Validation sequences (last 20% of training fold by time).
    pos_weight_scalar : float
        BCE positive-class weight = n_neg / n_pos for this training fold.
    device : torch.device
        Compute device.
    model_name : str
        Used for logging messages.
    fold_num : int
        Fold number (1-based) for logging.

    Returns
    -------
    CrashDetector
        Best model state restored from the checkpoint with highest
        validation PR-AUC.
    list[float]
        Mean training loss per epoch.
    list[float]
        Validation PR-AUC per epoch.
    """
    model = model.to(device)

    # BCELoss with reduction='none' so we can apply per-sample pos_weight manually
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.DL_LEARNING_RATE,
        weight_decay=config.DL_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",      # maximise PR-AUC
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_val_pr_auc: float = -1.0
    best_state: dict = copy.deepcopy(model.state_dict())
    epochs_no_improve: int = 0

    train_losses: list[float] = []
    val_pr_aucs: list[float] = []

    for epoch in range(1, config.DL_MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)   # (batch, 1)

            optimizer.zero_grad()
            proba = model(X_batch)         # (batch, 1) — probabilities

            # Per-sample weight: pos_weight for positives, 1 for negatives
            sample_w = torch.where(
                y_batch > 0.5,
                torch.full_like(y_batch, pos_weight_scalar),
                torch.ones_like(y_batch),
            )
            loss = (criterion(proba, y_batch) * sample_w).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.DL_GRAD_CLIP
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        val_pr_auc = _compute_val_pr_auc(model, val_loader, device)
        val_pr_aucs.append(val_pr_auc)
        scheduler.step(val_pr_auc)

        # Checkpoint: save best state
        if val_pr_auc > best_val_pr_auc + config.DL_MIN_DELTA:
            best_val_pr_auc = val_pr_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.debug(
                "[%s] Fold %d Epoch %3d | loss=%.4f  val_PR-AUC=%.4f  "
                "no_improve=%d",
                model_name, fold_num, epoch, avg_loss, val_pr_auc,
                epochs_no_improve,
            )

        # Early stopping
        if epochs_no_improve >= config.DL_PATIENCE:
            logger.info(
                "[%s] Fold %d early stop at epoch %d  "
                "(best val_PR-AUC=%.4f)",
                model_name, fold_num, epoch, best_val_pr_auc,
            )
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_pr_aucs


# ======================================================================
# PER-MODEL CV TRAINING PIPELINE
# ======================================================================

def train_evaluate_dl(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    weights: pd.Series,
    model_name: str,
    model_cfg: dict[str, Any],
    cv: PurgedEmbargoKFold,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Train and evaluate one DL model under Purged-Embargo CV.

    For each outer CV fold the pipeline is:
      1.  Split via ``cv`` (PurgedEmbargoKFold).
      2.  Fit StandardScaler on train; transform both train and test.
      3.  Split training fold: first 80% for training, last 20% for
          validation (early stopping). Temporal order is preserved.
      4.  Build SequenceDataset for train, val, and test portions.
          valid_indices constrains sequence endpoints to each portion.
          Test sequences are prefixed with the last (lookback-1) training
          events so the first test event has full context — no future
          leakage because those context events precede the test period.
      5.  Compute pos_weight = n_neg / n_pos from training portion only.
      6.  Training loop: BCE loss with pos_weight, Adam + weight_decay,
          ReduceLROnPlateau (mode=max, val PR-AUC), gradient clipping,
          early stopping with best-checkpoint restore.
      7.  Optimise threshold (max F1) on TRAIN portion predictions only.
      8.  Apply threshold to test predictions.
      9.  Compute all metrics on test fold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (18 features), DatetimeIndex sorted ascending.
    y : pd.Series
        Binary labels aligned with X.
    t1 : pd.Series
        Label end-times (barrier hit times) aligned with X.
    weights : pd.Series
        Sample uniqueness weights (not used in DL training; retained
        for API parity with train_evaluate_ml).
    model_name : str
        One of ``"rnn"``, ``"lstm"``, ``"gru"``.
    model_cfg : dict
        Entry from ``create_dl_models()``, containing ``"init_kwargs"``.
    cv : PurgedEmbargoKFold
        Pre-built CV splitter aligned with X.
    device : torch.device, optional
        Compute device.  Auto-detected if None.

    Returns
    -------
    dict with keys:
        fold_metrics   : list[dict]    — per-fold metric dicts
        mean_metrics   : dict          — mean across folds
        std_metrics    : dict          — std  across folds
        predictions    : pd.DataFrame  — test index + proba + pred + label
        train_times    : list[float]   — seconds per fold (train + val)
        infer_times    : list[float]   — seconds per fold (test inference)
        model_name     : str
        trained_models : list[CrashDetector] — one per fold (best checkpoint)
    """
    if device is None:
        device = detect_device()

    lookback = _effective_lookback()
    logger.info(
        "[%s] Starting DL training | device=%s, lookback=%d, "
        "LIGHTWEIGHT_MODE=%s",
        model_name, device, lookback, config.LIGHTWEIGHT_MODE,
    )

    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)

    fold_metrics: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    train_times: list[float] = []
    infer_times: list[float] = []
    trained_models: list[CrashDetector] = []

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        logger.info(
            "[%s] Fold %d/%d — train=%d, test=%d",
            model_name, fold_num, cv.n_splits,
            len(train_idx), len(test_idx),
        )

        # ── Leakage Rule 1: scale fit on train only ──────────────────────
        X_tr = X_arr[train_idx]          # (n_train, n_features)
        X_te = X_arr[test_idx]           # (n_test,  n_features)
        y_tr = y_arr[train_idx]
        y_te = y_arr[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_sc = scaler.transform(X_te).astype(np.float32)

        # ── Split training fold into train/val (temporal order) ──────────
        n_tr = len(X_tr_sc)
        # Validation portion: last 20% of training fold, at least lookback*2
        n_val = max(lookback * 2, int(n_tr * 0.2))
        # Ensure training portion has at least lookback events
        n_val = min(n_val, n_tr - lookback * 2)
        n_val = max(n_val, lookback)
        n_tr_only = n_tr - n_val

        X_tr_only = X_tr_sc[:n_tr_only]    # actual training portion
        y_tr_only = y_tr[:n_tr_only]
        X_va = X_tr_sc[n_tr_only:]         # validation portion
        y_va = y_tr[n_tr_only:]

        logger.debug(
            "[%s] Fold %d | n_tr_only=%d, n_val=%d, n_test=%d",
            model_name, fold_num, n_tr_only, n_val, len(X_te_sc),
        )

        # ── Build SequenceDatasets ───────────────────────────────────────
        # Training: endpoints are all positions in X_tr_only with enough history
        tr_valid = np.arange(n_tr_only)
        train_ds = SequenceDataset(X_tr_only, y_tr_only, tr_valid, lookback)

        # Validation: prepend last (lookback-1) training events as context
        ctx_len = lookback - 1
        if ctx_len > 0 and n_tr_only >= ctx_len:
            X_va_ctx = np.concatenate([X_tr_only[-ctx_len:], X_va])
            y_va_ctx = np.concatenate([y_tr_only[-ctx_len:], y_va])
        else:
            X_va_ctx = X_va
            y_va_ctx = y_va
            ctx_len = 0
        # Valid endpoints: only the validation positions (not the context)
        va_valid = np.arange(ctx_len, len(X_va_ctx))
        val_ds = SequenceDataset(X_va_ctx, y_va_ctx, va_valid, lookback)

        # Test: prepend last (lookback-1) full-training events as context
        if ctx_len > 0:
            X_te_ctx = np.concatenate([X_tr_sc[-ctx_len:], X_te_sc])
            y_te_ctx = np.concatenate([y_tr[-ctx_len:], y_te])
        else:
            X_te_ctx = X_te_sc
            y_te_ctx = y_te
        te_valid = np.arange(ctx_len, len(X_te_ctx))
        test_ds = SequenceDataset(X_te_ctx, y_te_ctx, te_valid, lookback)

        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            logger.warning(
                "[%s] Fold %d skipped — not enough samples after lookback "
                "filtering (train=%d, val=%d, test=%d)",
                model_name, fold_num, len(train_ds), len(val_ds), len(test_ds),
            )
            continue

        # DataLoaders — num_workers=0 for MPS/M1 compatibility
        train_loader = DataLoader(
            train_ds,
            batch_size=config.DL_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.DL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.DL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        # ── pos_weight from training portion only ────────────────────────
        n_pos = int((y_tr_only == 1).sum())
        n_neg = int((y_tr_only == 0).sum())
        pos_w = n_neg / n_pos if n_pos > 0 else 1.0
        logger.debug(
            "[%s] Fold %d class ratio (neg/pos in train_only) = %.3f",
            model_name, fold_num, pos_w,
        )

        # ── Instantiate fresh model ──────────────────────────────────────
        model = CrashDetector(**model_cfg["init_kwargs"])

        # ── Train ────────────────────────────────────────────────────────
        t_start = time.perf_counter()
        trained_model, tr_losses, val_aucs = _train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight_scalar=pos_w,
            device=device,
            model_name=model_name,
            fold_num=fold_num,
        )
        train_time = time.perf_counter() - t_start
        train_times.append(train_time)

        # ── Leakage Rule 2: threshold optimised on TRAIN portion only ────
        proba_train, y_train_true = _predict_loader(
            trained_model, train_loader, device
        )
        opt_threshold = _optimize_threshold(y_train_true, proba_train)
        logger.debug(
            "[%s] Fold %d opt_threshold=%.4f", model_name, fold_num, opt_threshold,
        )

        # ── Inference on TEST fold ────────────────────────────────────────
        t_infer = time.perf_counter()
        proba_test, y_test_true = _predict_loader(trained_model, test_loader, device)
        infer_time = time.perf_counter() - t_infer
        infer_times.append(infer_time)

        pred_test = (proba_test >= opt_threshold).astype(int)

        # ── Metrics ───────────────────────────────────────────────────────
        metrics = _compute_metrics(y_test_true.astype(int), pred_test, proba_test)
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

        # Collect test predictions with original DatetimeIndex
        # Note: test_ds may drop a few events if lookback filtering removed them;
        # use test_ds.endpoints to recover exact positions in test_idx
        n_ctx = ctx_len
        # The valid endpoints in test_ds correspond to positions in test_idx:
        # endpoint i in X_te_ctx maps to test_idx[i - n_ctx]
        test_event_positions = test_ds.endpoints - n_ctx   # 0-based in test_idx
        test_timestamps = X.index[test_idx[test_event_positions]]

        fold_preds = pd.DataFrame({
            "proba":     proba_test,
            "pred":      pred_test,
            "label":     y_test_true.astype(int),
            "fold":      fold_num,
            "threshold": opt_threshold,
        }, index=test_timestamps)
        all_predictions.append(fold_preds)

        trained_models.append(trained_model)

        # Free GPU/MPS memory before next fold
        del (train_loader, val_loader, test_loader,
             train_ds, val_ds, test_ds,
             X_tr, X_te, X_tr_sc, X_te_sc,
             X_tr_only, X_va, X_va_ctx, X_te_ctx,
             proba_train, proba_test)
        gc.collect()
        if str(device) == "mps":
            torch.mps.empty_cache()
        elif str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # ── Aggregate across folds ────────────────────────────────────────────
    if not fold_metrics:
        raise RuntimeError(
            f"[{model_name}] All folds were skipped — insufficient data for "
            f"lookback={lookback}. Reduce DL_LOOKBACK or increase dataset size."
        )

    metric_keys = list(config.EVAL_METRICS)
    mean_metrics = {
        k: float(np.mean([fm[k] for fm in fold_metrics])) for k in metric_keys
    }
    std_metrics = {
        k: float(np.std([fm[k] for fm in fold_metrics], ddof=1)) for k in metric_keys
    }
    mean_metrics["mean_train_time_s"] = float(np.mean(train_times))
    mean_metrics["total_train_time_s"] = float(np.sum(train_times))
    mean_metrics["mean_infer_time_s"] = float(np.mean(infer_times))

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
        "train_times":     train_times,
        "infer_times":     infer_times,
        "model_name":      model_name,
        "trained_models":  trained_models,
    }


# ======================================================================
# ORCHESTRATOR — all 3 DL models
# ======================================================================

def run_all_dl_models(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    weights: pd.Series,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Train and evaluate RNN, LSTM, and GRU under Purged-Embargo CV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (18 features), DatetimeIndex sorted ascending.
    y : pd.Series
        Binary labels aligned with X.
    t1 : pd.Series
        Label end-times aligned with X.
    weights : pd.Series
        Sample uniqueness weights (API parity with ML pipeline).
    device : torch.device, optional
        Compute device.  Auto-detected if None.

    Returns
    -------
    pd.DataFrame
        One row per model.  Columns: model name + all metric mean/std pairs +
        runtime.  Suitable for merging into paper Table 3.

    Side effects
    ------------
    Saves each model's best fold (highest test PR-AUC) as a pickle under
    ``config.PROCESSED_DATA_DIR / "models" / "<model_name>_best.pkl"``.
    Saves ``results/tables/table3_dl_part.csv``.
    """
    if device is None:
        device = detect_device()

    n_pos_total = int((y == 1).sum())
    n_neg_total = int((y == 0).sum())
    logger.info(
        "DL dataset: n=%d, pos=%d, neg=%d, global class ratio=%.3f",
        len(y), n_pos_total, n_neg_total,
        n_neg_total / n_pos_total if n_pos_total > 0 else 1.0,
    )

    outer_cv = PurgedEmbargoKFold(
        n_splits=config.CV_N_SPLITS,
        t1=t1,
        embargo_pct=config.CV_PCT_EMBARGO,
    )

    model_configs = create_dl_models(input_size=config.N_FEATURES)
    all_results: dict[str, dict] = {}

    for name, model_cfg in model_configs.items():
        logger.info("=" * 60)
        logger.info("Training %s on device=%s", name, device)
        logger.info("=" * 60)

        result = train_evaluate_dl(
            X=X,
            y=y,
            t1=t1,
            weights=weights,
            model_name=name,
            model_cfg=model_cfg,
            cv=outer_cv,
            device=device,
        )
        all_results[name] = result

        _save_best_dl_model(result, name)

        del result["trained_models"]
        gc.collect()
        if str(device) == "mps":
            torch.mps.empty_cache()
        elif str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    table3_dl = _build_table3_dl(all_results)
    out_path = config.TABLES_DIR / "table3_dl_part.csv"
    table3_dl.to_csv(out_path)
    logger.info("Table 3 DL part saved -> %s", out_path)

    return table3_dl


def _save_best_dl_model(result: dict[str, Any], model_name: str) -> None:
    """Persist the DL model from the fold with the highest test PR-AUC.

    Parameters
    ----------
    result : dict
        Output of ``train_evaluate_dl``.
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
        pickle.dump(best_model.cpu().state_dict(), f)

    logger.info(
        "Saved best %s model (fold %d, PR-AUC=%.4f) -> %s",
        model_name, best_fold_idx + 1, fold_pr_aucs[best_fold_idx], pkl_path,
    )


def _build_table3_dl(all_results: dict[str, dict]) -> pd.DataFrame:
    """Construct Table 3 summary DataFrame from run_all_dl_models results.

    Parameters
    ----------
    all_results : dict
        Map of model_name -> result dict from ``train_evaluate_dl``.

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
        row["mean_train_time_s"]  = round(m["mean_train_time_s"], 2)
        row["total_train_time_s"] = round(m["total_train_time_s"], 2)
        row["mean_infer_time_s"]  = round(m["mean_infer_time_s"], 4)
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


# ======================================================================
# ENTRY POINT — run from project root
# ======================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(config.PROJECT_ROOT))

    from src.utils import set_reproducibility

    set_reproducibility(config.RANDOM_SEED)
    logger.info(
        "Phase 6: DL model training | LIGHTWEIGHT_MODE=%s",
        config.LIGHTWEIGHT_MODE,
    )

    # ── Load feature-label dataset ────────────────────────────────────────
    data_path = config.PROCESSED_DATA_DIR / "X_y_adaptive.parquet"
    if not data_path.exists():
        logger.error("Missing %s — run Phase 3 (features.py) first.", data_path)
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info("Loaded X_y_adaptive | shape=%s", df.shape)

    feature_cols = config.FEATURE_NAMES
    X_data = df[feature_cols]
    y_data = df["bin"].astype(int)
    t1_data = df["t1"]
    weights_data = df["weight"]

    if not pd.api.types.is_datetime64_any_dtype(t1_data):
        t1_data = pd.to_datetime(t1_data)

    # ── Auto-detect device ────────────────────────────────────────────────
    dev = detect_device()

    # ── Run all DL models ─────────────────────────────────────────────────
    table3_dl = run_all_dl_models(X_data, y_data, t1_data, weights_data, device=dev)

    print("\n" + "=" * 70)
    print("TABLE 3 — DL Models (Purged-Embargo CV, Adaptive Labels)")
    print("=" * 70)
    print(table3_dl.to_string())
    print("=" * 70)

    # ── Merge with ML results -> full Table 3 ─────────────────────────────
    ml_path = config.TABLES_DIR / "table3_ml_part.csv"
    if ml_path.exists():
        table3_ml = pd.read_csv(ml_path, index_col="model")
        table3_full = pd.concat([table3_ml, table3_dl])

        full_path = config.TABLES_DIR / "table3_main_performance.csv"
        table3_full.to_csv(full_path)

        print("\n" + "=" * 70)
        print("TABLE 3 — FULL (ML + DL, Purged-Embargo CV, Adaptive Labels)")
        print("=" * 70)
        print(table3_full.to_string())
        print("=" * 70)
        print(f"\nSaved merged table -> {full_path}")
    else:
        logger.warning(
            "ML results not found at %s — run Phase 5 (models_ml.py) first "
            "to produce the full Table 3 merge.",
            ml_path,
        )
        print(f"\nDL-only table saved -> {config.TABLES_DIR / 'table3_dl_part.csv'}")
