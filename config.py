"""
config.py — Single source of truth for ALL project parameters.
Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection
"""

from pathlib import Path

# ======================================================================
# RUNTIME MODE — MEMORY CONSTRAINTS
# ======================================================================
LIGHTWEIGHT_MODE: bool = True    # Set False when running on HPC
SAMPLE_ROWS: int = 200_000       # Use last N rows in lightweight mode
USE_FLOAT32: bool = True         # Convert float64 → float32 to halve memory

# ======================================================================
# PATHS
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_DATA_FILE = RAW_DATA_DIR / "EURUSD_1.csv"

# Ensure output directories exist at import time
for _dir in [PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ======================================================================
# REPRODUCIBILITY
# ======================================================================
RANDOM_SEED: int = 42

# ======================================================================
# DATA PREPARATION
# ======================================================================
# Expected columns in raw CSV (OHLCV)
RAW_COLUMNS: list[str] = ["datetime", "open", "high", "low", "close", "volume"]
DATETIME_COL: str = "datetime"
FREQ: str = "1min"  # 1-minute bars

# ======================================================================
# CUSUM FILTER
# ======================================================================
CUSUM_THRESHOLD_MULTIPLIER: float = 2.0   # threshold = multiplier * rolling_std
CUSUM_LOOKBACK: int = 50                  # bars for rolling volatility estimate
# Target ~7% event rate (from sensitivity analysis)
CUSUM_MIN_RETURN: float = 0.0             # minimum |return| to trigger event

# ======================================================================
# TRIPLE BARRIER METHOD — STANDARD
# ======================================================================
TBM_PROFIT_TAKING: float = 1.5           # PT multiplier × daily volatility
TBM_STOP_LOSS: float = 1.0               # SL multiplier × daily volatility
TBM_HORIZON_BARS: int = 60               # max holding period in bars (= 60 min)
TBM_VOLATILITY_SPAN: int = 50            # EWM span for daily vol estimate
# Label encoding: 1=crash(PT hit), 0=no-crash (SL or timeout)
CRASH_LABEL: int = 1
NO_CRASH_LABEL: int = 0

# ======================================================================
# TRIPLE BARRIER METHOD — ADAPTIVE
# ======================================================================
# Adaptive barriers scale with local volatility regime
TBM_ADAPTIVE_PT_LOW_VOL: float = 2.0     # PT multiplier in low-vol regime
TBM_ADAPTIVE_PT_HIGH_VOL: float = 1.0    # PT multiplier in high-vol regime
TBM_ADAPTIVE_SL_LOW_VOL: float = 1.5     # SL multiplier in low-vol regime
TBM_ADAPTIVE_SL_HIGH_VOL: float = 0.75   # SL multiplier in high-vol regime
TBM_ADAPTIVE_VOL_LOOKBACK: int = 100     # bars for regime vol estimate
TBM_ADAPTIVE_VOL_PERCENTILE: float = 0.75  # percentile to define high-vol regime

# ======================================================================
# FEATURE ENGINEERING
# ======================================================================
# Group 1 — Volatility & Range
VOLATILITY_EWM_SPAN: int = 50            # EWM span for log-return std
GK_VOL_WINDOW: int = 20                  # Garman-Klass rolling mean window
HIGH_LOW_WINDOW: int = 20                # high-low range rolling mean window
BB_WINDOW: int = 20                      # Bollinger Band MA window
BB_STD_MULTIPLIER: float = 2.0           # Bollinger Band std multiplier
ROLL_SKEW_WINDOW: int = 60               # rolling skewness window (bars)
ROLL_KURT_WINDOW: int = 60               # rolling kurtosis window (bars)

# Group 2 — Microstructure Proxies
AMIHUD_EPSILON: float = 1e-9             # small constant to avoid division by zero
CS_SPREAD_WINDOW: int = 20               # Corwin-Schultz rolling window
VOLUME_MA_WINDOW: int = 20               # volume rolling mean window

# Group 3 — Momentum & Return
RSI_PERIOD: int = 14                     # RSI period (bars)
EFFICIENCY_RATIO_WINDOW: int = 20        # fractal efficiency ratio window

# Group 4 — Trend & Seasonality
EMA_WINDOW: int = 20                     # EMA window for deviation feature
VWAP_WINDOW: int = 20                    # VWAP rolling window
MINUTES_IN_DAY: int = 1440               # for time encoding (sin/cos)

# Full ordered feature list (18 features)
FEATURE_NAMES: list[str] = [
    # Group 1 — Volatility & Range
    "volatility",
    "garman_klass_vol",
    "high_low_range",
    "bb_width",
    "roll_skew",
    "roll_kurt",
    # Group 2 — Microstructure Proxies
    "amihud",
    "cs_spread",
    "volume_change_ratio",
    "interact_vol_amihud",
    # Group 3 — Momentum & Return
    "log_return",
    "return_lag_1",
    "rsi_14",
    "efficiency_ratio",
    # Group 4 — Trend & Seasonality
    "ema_deviation",
    "vwap_deviation",
    "time_sin",
    "time_cos",
]
N_FEATURES: int = len(FEATURE_NAMES)  # must equal 18

# ======================================================================
# SAMPLE WEIGHTS
# ======================================================================
# Concurrency-based uniqueness weights (López de Prado, Chapter 4)
SAMPLE_WEIGHT_DECAY: float = 0.0         # 0 = no time decay, >0 = exponential decay
SAMPLE_WEIGHT_MIN: float = 0.0           # floor for normalized weights

# ======================================================================
# PURGED-EMBARGO CROSS-VALIDATION
# ======================================================================
CV_N_SPLITS: int = 5                     # number of CV folds
CV_PCT_EMBARGO: float = 0.01             # embargo as fraction of total time span
# Note: embargo is TIME-BASED (see Bug #1 fix in CLAUDE.md)

# ======================================================================
# ML MODEL PARAMETERS — LightGBM
# ======================================================================
LGBM_PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# ======================================================================
# ML MODEL PARAMETERS — XGBoost
# ======================================================================
XGBOOST_PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    # scale_pos_weight computed dynamically per fold (n_neg / n_pos)
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

# ======================================================================
# ML MODEL PARAMETERS — Random Forest
# ======================================================================
RF_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ======================================================================
# DL MODEL PARAMETERS — RNN / LSTM / GRU (PyTorch)
# ======================================================================
DL_LOOKBACK: int = 20                    # sequence length (bars); stays within fold
DL_HIDDEN_SIZE: int = 64                 # hidden units per recurrent layer
DL_NUM_LAYERS: int = 2                   # stacked recurrent layers
DL_DROPOUT: float = 0.3                  # dropout between recurrent layers
DL_BATCH_SIZE: int = 64
DL_LEARNING_RATE: float = 1e-3
DL_WEIGHT_DECAY: float = 1e-4           # L2 regularization
DL_MAX_EPOCHS: int = 100
DL_PATIENCE: int = 10                    # early stopping patience
DL_MIN_DELTA: float = 1e-4              # min improvement for early stopping
DL_GRAD_CLIP: float = 1.0               # gradient clipping max norm
DL_DEVICE: str = "cpu"                   # "cuda" if GPU available; set at runtime

# ======================================================================
# EVALUATION
# ======================================================================
# Metrics to compute (all reported in Table 3)
EVAL_METRICS: list[str] = [
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "brier_score",
    "mcc",
]
# Threshold optimization target metric (inside each CV fold)
THRESHOLD_OPT_METRIC: str = "f1"        # optimize F1 on training fold
THRESHOLD_SEARCH_GRID: int = 100        # number of threshold candidates [0,1]
# Note: NEVER use hardcoded threshold 0.4658 (Bug #4 in CLAUDE.md)

# ======================================================================
# HYPERPARAMETER TUNING (RandomizedSearchCV)
# ======================================================================
TUNING_N_ITER: int = 20               # number of random parameter settings sampled
TUNING_SCORING: str = "average_precision"  # optimize PR-AUC during tuning

ML_TUNING_SPACE: dict = {
    "lgbm": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "xgboost": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "rf": {
        "n_estimators": [200, 300, 500],
        "max_depth": [6, 10, 15, None],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    },
}

# ======================================================================
# ROBUSTNESS CHECKS (Table 8)
# ======================================================================
ROBUSTNESS_HORIZONS: list[int] = [30, 60, 90]          # alternative TBM horizons (bars)
ROBUSTNESS_PT_MULTIPLIERS: list[float] = [1.0, 1.5, 2.0]  # alternative PT barriers
ROBUSTNESS_SL_MULTIPLIERS: list[float] = [0.75, 1.0, 1.5]  # alternative SL barriers
ROBUSTNESS_EMBARGO_PCTS: list[float] = [0.005, 0.01, 0.02]  # alternative embargo %
ROBUSTNESS_FEATURE_SUBSETS: list[str] = [
    "all",           # baseline: all 18 features
    "no_micro",      # drop Group 2 microstructure features
    "vol_only",      # Group 1 only
    "momentum_only", # Group 3 only
]

# ======================================================================
# SHAP / INTERPRETABILITY
# ======================================================================
SHAP_BACKGROUND_SAMPLES: int = 100      # TreeExplainer background sample size
SHAP_MAX_DISPLAY: int = 18              # features to show in summary plot
IG_N_STEPS: int = 50                    # Integrated Gradients approximation steps
IG_BASELINE: str = "zero"               # baseline type: "zero" or "mean"

# ======================================================================
# PLOTTING
# ======================================================================
FIGURE_DPI: int = 300
FIGURE_FORMAT: str = "pdf"              # publication format
FIGURE_STYLE: str = "seaborn-v0_8-whitegrid"
FIGURE_PALETTE: str = "colorblind"      # accessible color palette
FIGURE_FONT_SIZE: int = 11
FIGURE_TITLE_SIZE: int = 13
FIGURE_LABEL_SIZE: int = 11
FIGURE_TICK_SIZE: int = 9
FIGURE_LEGEND_SIZE: int = 9
# Model display names (consistent across all figures/tables)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "lgbm": "LightGBM",
    "xgboost": "XGBoost",
    "rf": "Random Forest",
    "rnn": "RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}
MODEL_COLORS: dict[str, str] = {
    "lgbm": "#0072B2",
    "xgboost": "#E69F00",
    "rf": "#009E73",
    "rnn": "#CC79A7",
    "lstm": "#56B4E9",
    "gru": "#D55E00",
}
