# ====================================================================
# CLAUDE.md — Project Context & Coding Standards
# Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection
# ====================================================================

## Project Identity

This is a master's thesis project implementing a leakage-aware machine
learning framework for detecting flash-crash-like events in 1-minute
EUR/USD OHLCV data. The paper compares 6 models (LightGBM, XGBoost,
Random Forest, RNN, LSTM, GRU) under Purged-Embargo Cross-Validation
with adaptive Triple Barrier labeling.

The paper has 5 research questions (RQ1–RQ5), 8 tables, and 10 figures
that must be filled with experimental results.

## Critical Domain Knowledge

### What is a flash crash in this context?
A short-duration extreme price dislocation in FX markets caused by
liquidity withdrawal, algorithmic trading reactions, and microstructure
fragility — NOT ordinary high volatility. Detection = rare-event
binary classification (crash vs no-crash).

### Why "leakage-aware"?
Triple Barrier labels use FUTURE price paths (up to 60 min ahead).
If training data contains events whose label horizons overlap with
test data, the model sees future information → inflated performance.
Purged-Embargo CV removes this overlap. This is the methodological
backbone of the paper — if validation is wrong, ALL results are invalid.

### What makes this different from standard ML?
1. Labels are path-dependent (not fixed-horizon returns)
2. Events are irregularly spaced (CUSUM-triggered, not every bar)
3. Extreme class imbalance (crashes < 10% of events)
4. Temporal dependence requires specialized CV

## Directory Structure

```
project/
├── CLAUDE.md              ← YOU ARE HERE
├── config.py              ← Single source of truth for ALL parameters
├── requirements.txt       ← Pinned Python dependencies
├── data/
│   ├── raw/               ← EURUSD_1.csv (NEVER modify)
│   └── processed/         ← All intermediate outputs (parquet format)
├── src/
│   ├── __init__.py
│   ├── data_prep.py       ← Module 1: Loading, cleaning
│   ├── cusum.py           ← Module 2: CUSUM event filter
│   ├── labeling.py        ← Module 3: Standard + Adaptive TBM
│   ├── features.py        ← Module 4: Feature engineering (18 features, 4 groups)
│   ├── sample_weights.py  ← Module 5: Concurrency-based uniqueness weights
│   ├── purged_cv.py       ← Module 6: Purged-Embargo CV ⚠️ CRITICAL
│   ├── models_ml.py       ← Module 7: LightGBM, XGBoost, RF
│   ├── models_dl.py       ← Module 8: RNN, LSTM, GRU (PyTorch)
│   ├── evaluation.py      ← Module 9: All metrics + threshold optimization
│   ├── interpretability.py ← Module 10: SHAP + Integrated Gradients
│   ├── plotting.py        ← Module 11: Publication figures
│   └── utils.py           ← Module 12: Logging, timing, seeds
├── notebooks/             ← Orchestration only (import from src/)
├── results/
│   ├── tables/            ← CSV outputs for Tables 1–8
│   └── figures/           ← PDF/PNG for Figures 1–10
└── tests/                 ← Unit tests for critical modules
```

## Coding Standards (MUST follow)

1. **Type hints** on ALL function signatures
2. **Docstrings** on ALL public functions (purpose, params, returns)
3. **Logging** instead of print statements (use `logging` module)
4. **All parameters from config.py** — NEVER hardcode values
5. **Save intermediates as parquet** (preserves DatetimeIndex + dtypes)
6. **Relative paths only** via config.PROJECT_ROOT
7. **Random seed** from config.RANDOM_SEED everywhere
8. **No experiment logic in notebooks** — notebooks only import from src/
9. **Every function must be testable** in isolation

## Leakage Prevention Rules (NON-NEGOTIABLE)

These rules override all other considerations. Violating any of them
invalidates the entire experiment:

1. **Feature scaling INSIDE each CV fold** — fit scaler on train only,
   transform test using the training scaler. NEVER fit on full dataset.

2. **Threshold optimization INSIDE each CV fold** — find optimal
   classification threshold using only training data within the fold.
   NEVER use a global threshold like the hardcoded 0.4658 from old code.

3. **Purged-Embargo CV for ALL model evaluation** — training events
   whose label horizon [event_time, t1] overlaps ANY part of the
   test period must be removed. Embargo is TIME-BASED, not index-based.

4. **Hyperparameter tuning INSIDE CV** — nested CV or tuning only on
   training folds. NEVER tune on full dataset then evaluate.

5. **DL sequence construction respects fold boundaries** — no lookback
   sequence may reach into purged/embargoed regions or cross fold
   boundaries.

6. **No future features** — all features use only past/current data
   relative to the event time. Rolling windows look backward only.

## Feature Engineering Specification (18 features, 4 groups)

All features justified by literature review citations:

**Group 1 — Volatility & Range (6 features):**
- `volatility`: EWM std of log_return, span=50
  → Bollerslev et al. 2018; Ardia et al. 2018
- `garman_klass_vol`: 0.5*log(H/L)^2 - (2*ln2-1)*log(C/O)^2, rolling mean 20
  → Fiszeder et al. 2019 (range-based outperforms return-based)
- `high_low_range`: (high-low)/close, rolling mean 20
  → Fiszeder et al. 2019
- `bb_width`: Bollinger Band width = (upper-lower)/MA_20
  → Christensen et al. 2025
- `roll_skew`: rolling skewness of log_return, 60-bar window
  → Christensen et al. 2025 (tail risk signals)
- `roll_kurt`: rolling kurtosis of log_return, 60-bar window
  → Christensen et al. 2025

**Group 2 — Microstructure Proxies (4 features):**
- `amihud`: |log_return| / (volume * close + 1e-9)
  → Ranaldo & Somogyi 2021 (liquidity withdrawal pre-crash)
- `cs_spread`: Corwin-Schultz bid-ask spread estimator
  → Cenedese et al. 2021 (spread widens 30-50bps under stress)
- `volume_change_ratio`: volume / rolling_mean_volume(20) - 1
  → Piccotti & Schreiber 2020 (volume dynamics in jumps)
- `interact_vol_amihud`: volatility * amihud
  → Breedon et al. 2023 (high vol + low liquidity = stress)

**Group 3 — Momentum & Return (4 features):**
- `log_return`: log(close/close.shift(1))
  → Gu et al. 2020
- `return_lag_1`: log_return.shift(1)
  → Christensen et al. 2025 (lagged returns matter)
- `rsi_14`: Relative Strength Index, 14-bar
  → standard momentum indicator
- `efficiency_ratio`: |price_change(20)| / sum(|bar_changes(20)|)
  → fractal efficiency measure

**Group 4 — Trend & Seasonality (4 features):**
- `ema_deviation`: (close - EMA_20) / EMA_20
  → paper Section 4.4
- `vwap_deviation`: (close - VWAP_20) / VWAP_20
  → paper Section 4.4
- `time_sin`: sin(2π * minutes_in_day / 1440)
  → Hasbrouck & Levich 2021 (FX activity concentration)
- `time_cos`: cos(2π * minutes_in_day / 1440)
  → Hasbrouck & Levich 2021

## Data Summary (Phase 1 Results)

| Item | Value |
|------|-------|
| Raw file rows | 2,207,595 |
| Date range | 2020-03-12 to 2025-03-31 (~5 years) |
| Cleaned bars | 2,207,594 (1 row dropped — leading NaN log_return) |
| Duplicate timestamps removed | 0 |
| Zero-volume bars removed | 0 |
| CUSUM events (m=2.0) | 339,098 (15.36% of valid bars) |
| ADF p-value (log returns) | 0.000000 (stationary) |
| log_return skewness | −210.02 |
| log_return kurtosis | 163,219 |
| Processed outputs | `data/processed/df_clean.parquet`, `data/processed/cusum_events.parquet` |

## Known Bugs in Legacy Code (MUST fix)

### Bug 1: PurgedKFold embargo is index-based (WRONG)
```python
# OLD (WRONG): embargo = int(n * self.pct_embargo)
# CORRECT: compute time-based embargo duration
total_span = X.index[-1] - X.index[0]
embargo_td = total_span * self.pct_embargo
```

### Bug 2: PurgedKFold deletes training data BEFORE test fold
```python
# OLD (WRONG): train_idx = train_idx[train_idx > cutoff]
# This removes ALL training data before the test fold!
# CORRECT: Only remove embargo zone AFTER the test fold
```

### Bug 3: Purging logic incomplete
```python
# OLD (WRONG): checks only if training event's END falls in test
# CORRECT: check FULL overlap
purge_mask = (t1.iloc[train_idx] > test_start) & \
             (X.index[train_idx] < test_end)
```

### Bug 4: Hardcoded threshold 0.4658
Threshold must be optimized per fold, not fixed globally.

### Bug 5: XGBoost scale_pos_weight=1
Must be computed dynamically: n_negative / n_positive per fold.

### Bug 6: Absolute file paths
Change from /Users/ismathakit/... to relative paths via config.py.

### Known Issue: CUSUM_THRESHOLD_MULTIPLIER mismatch
`CUSUM_THRESHOLD_MULTIPLIER = 2.0` in config.py was based on an original
estimate of ~7% event rate, but on the full 5-year dataset m=2.0 yields
15.36%. Multiplier 3.0 gives 7.74% which is closest to the target.
Consider updating to 3.0, or testing both values as part of robustness
checks (Table 8). Current experiments use m=2.0 unless overridden.

## Research Questions → Code Mapping

| RQ | Question | Code Location | Output |
|----|----------|---------------|--------|
| RQ1 | What dataset is produced? | notebooks/02 | Table 1, Table 2 |
| RQ2 | How do models perform under purged CV? | notebooks/04+05 | Table 3, Fig 2-3 |
| RQ3 | Does adaptive labeling improve quality? | notebooks/06 | Table 4, Fig 4-5 |
| RQ4 | How much leakage inflation? | notebooks/07 | Table 5, Fig 6 |
| RQ5 | Interpretability + robustness? | notebooks/08+09 | Table 6-8, Fig 7-10 |

## Paper Deliverables Checklist

- [ ] Table 1: Dataset summary (bars, events, horizon, embargo, period)
- [ ] Table 2: Class distribution (crash/no-crash per split)
- [ ] Table 3: Main performance (6 models × 7 metrics + runtime)
- [ ] Table 4: Standard vs adaptive labeling comparison
- [ ] Table 5: Purged-Embargo vs standard CV comparison
- [ ] Table 6: Top features per model family
- [ ] Table 7: Computational feasibility (training/inference time)
- [ ] Table 8: Robustness checks (alt horizon, barrier, features, embargo)
- [ ] Figure 2: PR curves for all 6 models
- [ ] Figure 3: Model ranking comparison
- [ ] Figure 4: Class distribution under standard vs adaptive
- [ ] Figure 5: Performance difference (adaptive - standard)
- [ ] Figure 6: Leakage inflation visualization
- [ ] Figure 7: SHAP summary (best tree-based model)
- [ ] Figure 8: Integrated Gradients summary (best recurrent model)
- [ ] Figure 9: Local explanation for representative crash event
- [ ] Figure 10: Sensitivity analysis across settings

## Progress Log
<!-- Update this after each Claude Code session -->

- [x] Phase 0: Project structure + config.py
- [x] Phase 1: data_prep.py + cusum.py
- [ ] Phase 2: labeling.py (standard + adaptive TBM)
- [ ] Phase 3: features.py + sample_weights.py
- [ ] Phase 4: purged_cv.py (FIXED) + tests
- [ ] Phase 5: models_ml.py (LightGBM, XGBoost, RF)
- [ ] Phase 6: models_dl.py (RNN, LSTM, GRU)
- [ ] Phase 7: evaluation.py + comparison experiments
- [ ] Phase 8: interpretability.py (SHAP + IG)
- [ ] Phase 9: robustness checks
- [ ] Phase 10: Final figures + tables

## Key Decisions Log
<!-- Record decisions that affect multiple modules -->

| Date | Decision | Rationale |
|------|----------|-----------|
| | CUSUM threshold = 2.0 * sigma | Sensitivity analysis showed ~7% event rate |
| | TBM: PT=1.5, SL=1.0, horizon=60min | Balance between sensitivity and label quality |
| | 18 features, 4 groups | All justified by literature review |
| | DL lookback = 20 bars | Short enough to stay within fold boundaries |
| 2026-04-09 | Raw CSV has 7 columns, last column is flag/trade column to be dropped | Discovered during Phase 0 setup; data_prep.py must assign column names manually and drop col index 6 |
| 2026-04-09 | Raw data: 2,207,594 bars, date range 2020-03-12 to 2025-03-31 (~5 years) | Established in Phase 1; drives fold sizing, embargo duration, and expected event counts |
| 2026-04-09 | CUSUM multiplier 2.0 gives 15.36% event rate (339,098 events), NOT ~7% as originally estimated. Multiplier 3.0 gives 7.74% (170,851 events). Decision: keep m=2.0 for now — 15% is still reasonable for initial experiments. Can adjust later in robustness checks (Table 8). | Original ~7% estimate was based on a smaller dataset; full 5-year dataset shows higher base rate |
| 2026-04-09 | Log return shows extreme skew (-210) and kurtosis (163,219) — confirms fat tails and crash signatures in the data | Validates the motivation for the flash-crash detection task; standard Gaussian assumptions would be severely violated |