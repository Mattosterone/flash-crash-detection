"""
src — Flash Crash Detection package.

Modules
-------
data_prep       : Module 1 — data loading and cleaning
cusum           : Module 2 — CUSUM event filter
labeling        : Module 3 — standard and adaptive Triple Barrier Method
features        : Module 4 — feature engineering (18 features, 4 groups)
sample_weights  : Module 5 — concurrency-based uniqueness weights
purged_cv       : Module 6 — Purged-Embargo cross-validation
models_ml       : Module 7 — LightGBM, XGBoost, Random Forest
models_dl       : Module 8 — RNN, LSTM, GRU (PyTorch)
evaluation      : Module 9 — metrics and threshold optimization
interpretability: Module 10 — SHAP and Integrated Gradients
plotting        : Module 11 — publication figures
utils           : Module 12 — logging, timing, seeds
"""
