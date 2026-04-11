# Flash Crash Detection — Leakage-Aware ML Framework

**Master's Thesis Project**
Binary classification of flash-crash-like events in 1-minute EUR/USD OHLCV data using a leakage-aware machine learning framework with Purged-Embargo Cross-Validation and adaptive Triple Barrier labeling.

---

## Overview

Flash crashes in FX markets are short-duration extreme price dislocations caused by liquidity withdrawal and algorithmic feedback loops — not ordinary volatility. Detecting them requires rare-event classification with strict safeguards against temporal data leakage.

This project addresses three methodological challenges common in financial ML:

1. **Label leakage** — Triple Barrier labels use future price paths up to 60 bars ahead; standard CV exposes future data to models
2. **Feature leakage** — scalers and thresholds fitted on full dataset before splitting inflate performance
3. **Class imbalance** — crashes are a rare minority class requiring adapted evaluation metrics

The framework enforces leakage prevention at every stage and compares 6 models (LightGBM, XGBoost, Random Forest, RNN, LSTM, GRU) under identical conditions.

---

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | What are the statistical properties of the labeled dataset? |
| RQ2 | How do the 6 models perform under purged-embargo CV? |
| RQ3 | Does adaptive barrier labeling improve label quality vs. standard TBM? |
| RQ4 | How much performance inflation does standard CV introduce vs. purged CV? |
| RQ5 | Which features drive predictions and how robust are results? |

---

## Dataset

| Property | Value |
|----------|-------|
| Instrument | EUR/USD spot |
| Frequency | 1-minute OHLCV bars |
| Period | 2020-03-12 to 2025-03-31 (~5 years) |
| Raw rows | 2,207,595 |
| Cleaned bars | 2,207,594 |
| CUSUM events (m=2.0) | 339,098 (15.36% of bars) |
| Label horizon | 60-minute max holding period |
| Task | Binary classification: crash (1) vs. no-crash (0) |

Data file: `data/raw/EURUSD_1.csv` (never modified; all outputs written to `data/processed/`).

---

## Methodology

### Event Filtering — CUSUM

Symmetric CUSUM filter with threshold = 2.0 × rolling volatility (50-bar EWM std) generates candidate events. Only CUSUM-triggered bars are labeled and used for model training.

### Labeling — Triple Barrier Method (TBM)

Two variants:
- **Standard TBM**: fixed profit-taking (1.5×) and stop-loss (1.0×) barriers relative to local volatility
- **Adaptive TBM**: barriers scale with volatility regime (wider in low-vol, tighter in high-vol)

Label = 1 (crash) if profit-taking barrier is hit first; 0 otherwise.

### Feature Engineering — 18 Features, 4 Groups

| Group | Features |
|-------|----------|
| Volatility & Range | `volatility`, `garman_klass_vol`, `high_low_range`, `bb_width`, `roll_skew`, `roll_kurt` |
| Microstructure Proxies | `amihud`, `cs_spread`, `volume_change_ratio`, `interact_vol_amihud` |
| Momentum & Return | `log_return`, `return_lag_1`, `rsi_14`, `efficiency_ratio` |
| Trend & Seasonality | `ema_deviation`, `vwap_deviation`, `time_sin`, `time_cos` |

All features are strictly backward-looking relative to event time.

### Cross-Validation — Purged-Embargo KFold

Standard K-Fold is invalid for path-dependent labels. This project implements `PurgedEmbargoKFold` which:

- **Purges** training events whose label horizon `[event_time, t1]` overlaps the test window
- **Embargoes** a time buffer after the test fold (1% of total span ≈ 18 days) to prevent look-ahead from lagged features
- All embargo durations are **time-based**, not index-based

Three legacy bugs in common implementations are corrected (see `src/purged_cv.py`).

### Leakage Prevention Rules

All of these are enforced inside each fold, never globally:

- Feature scaling: scaler fitted on training split only
- Threshold optimization: classification threshold searched on training split
- Hyperparameter tuning: nested CV (inner `PurgedEmbargoKFold`) within training fold
- DL sequences: no lookback window crosses fold boundaries

### Models

**ML (scikit-learn / LightGBM / XGBoost):** Each model uses nested CV for hyperparameter tuning. Class imbalance handled via `class_weight='balanced'` (LightGBM, RF) or dynamic `scale_pos_weight` (XGBoost).

**DL (PyTorch):** RNN, LSTM, GRU with sequence length 20. Validation split = last 20% of each training fold by time. Early stopping + ReduceLROnPlateau. Apple M1 MPS backend supported.

---

## Results Summary (Lightweight Mode — 200K rows)

> These numbers come from the development subset (last 200K bars). Full-dataset results for the paper are produced on HPC with `LIGHTWEIGHT_MODE=False`.

**Table 3 — Main Performance (5-fold Purged-Embargo CV):**

| Model | PR-AUC | MCC |
|-------|--------|-----|
| LightGBM | 0.632 ± 0.024 | ~0.09 |
| XGBoost | 0.635 ± 0.024 | ~0.09 |
| Random Forest | 0.634 ± 0.024 | 0.093 ± 0.028 |
| RNN | ~0.54 | — |
| LSTM | 0.627 ± 0.002 | — |
| GRU | ~0.54 | — |

**Top features (RF Gini importance):** `bb_width`, `log_return`, `volatility`, `ema_deviation`, `vwap_deviation`

**Robustness:** Embargo sensitivity ±2× changes PR-AUC by < 0.001. Narrow barriers (pt=1.0, sl=0.5) give best MCC=0.126.

**Leakage check:** Purged CV vs. standard CV inflation near-zero on 200K subset (expected; full dataset will show larger gap).

---

## Project Structure

```
flash-crash-detection/
├── config.py                   # All parameters — single source of truth
├── requirements.txt
├── run_all.py                  # Sequential pipeline runner
├── generate_remaining.py       # Final tables + figures (Phase 10)
├── data/
│   ├── raw/EURUSD_1.csv        # Never modified
│   └── processed/              # Parquet intermediates
├── src/
│   ├── data_prep.py            # Loading, cleaning
│   ├── cusum.py                # CUSUM event filter
│   ├── labeling.py             # Standard + Adaptive TBM
│   ├── features.py             # 18 features, 4 groups
│   ├── sample_weights.py       # Concurrency-based uniqueness weights
│   ├── purged_cv.py            # PurgedEmbargoKFold (leakage-safe CV)
│   ├── models_ml.py            # LightGBM, XGBoost, Random Forest
│   ├── models_dl.py            # RNN, LSTM, GRU (PyTorch)
│   ├── evaluation.py           # Metrics + threshold optimization
│   ├── interpretability.py     # SHAP + Integrated Gradients
│   ├── robustness.py           # Table 8 sensitivity checks
│   ├── pipeline_runner.py      # Phase orchestration
│   └── utils.py                # Logging, seeds, timing
├── notebooks/
│   └── 10_final.ipynb          # Final results notebook
├── results/
│   ├── tables/                 # table1.csv … table8.csv
│   └── figures/                # figure2.pdf … figure10.pdf
└── tests/                      # Unit tests (21 tests for purged_cv)
```

---

## Installation

```bash
# Python 3.10+
pip install -r requirements.txt
```

PyTorch with MPS (Apple Silicon):
```bash
pip install torch>=2.0  # MPS backend is included automatically
```

---

## Running the Pipeline

### Full sequential run (all phases):
```bash
python run_all.py
```

### Individual phases:
```bash
# Phase 1: Data prep + CUSUM
python -c "from src.pipeline_runner import run_phase; run_phase(1)"

# Phase 2: Labeling
python -c "from src.pipeline_runner import run_phase; run_phase(2)"

# ... phases 3–9 follow the same pattern

# Phase 10: Generate final tables + figures
python generate_remaining.py
```

### Run tests:
```bash
python -m pytest tests/ -v
```

---

## Configuration

All parameters are in `config.py`. Key switches:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LIGHTWEIGHT_MODE` | `True` | Use last 200K rows (M1 8GB development mode) |
| `RANDOM_SEED` | `42` | Global reproducibility seed |
| `CUSUM_THRESHOLD_MULTIPLIER` | `2.0` | Event filter sensitivity |
| `TBM_PROFIT_TAKING` | `1.5` | Crash barrier multiplier |
| `TBM_HORIZON_BARS` | `60` | Max label horizon (minutes) |
| `CV_N_SPLITS` | `5` | Number of purged CV folds |
| `EMBARGO_PCT` | `0.01` | Embargo as fraction of total time span |

**For paper results:** set `LIGHTWEIGHT_MODE = False` and run on HPC.

---

## Paper Deliverables Status

| # | Deliverable | Status |
|---|-------------|--------|
| Table 1 | Dataset summary | Done |
| Table 2 | Class distribution per fold | Done |
| Table 3 | Main performance (6 models × 7 metrics) | Done |
| Table 4 | Standard vs. adaptive labeling | Done |
| Table 5 | Purged vs. standard CV leakage comparison | Done |
| Table 6 | Top features per model family | Done |
| Table 7 | Computational feasibility | Done |
| Table 8 | Robustness checks (8 settings) | Done |
| Figures 2–10 | All publication figures | Done |

> All outputs in `results/tables/` and `results/figures/`. Final HPC run needed to replace lightweight-mode numbers with full-dataset paper numbers.

---

## Hardware Notes

Developed on MacBook Air M1 (8GB RAM). `LIGHTWEIGHT_MODE=True` keeps peak memory under 4GB. Final paper results require HPC with `LIGHTWEIGHT_MODE=False` (full 2.2M bar dataset, ~30GB RAM recommended).

MPS acceleration is used automatically for PyTorch models when running on Apple Silicon.

---

## Key References

- Bollerslev et al. (2018) — volatility estimation
- López de Prado (2018) — *Advances in Financial Machine Learning* (TBM, Purged CV)
- Christensen et al. (2025) — tail-risk features for crash detection
- Ranaldo & Somogyi (2021) — Amihud illiquidity in FX
- Cenedese et al. (2021) — bid-ask spread under stress (Corwin-Schultz)
- Hasbrouck & Levich (2021) — intraday seasonality in FX
- Breedon et al. (2023) — liquidity-volatility interaction

---

## License

Academic use only. Data sourced from proprietary FX feed — not redistributable.
