"""
run_all.py — End-to-end pipeline runner.
Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection

Executes all phases sequentially with explicit memory cleanup between steps.
Designed for both lightweight (M1 Air 8 GB) and full HPC runs.

Usage
-----
# Lightweight mode (fast, development; last 200K rows):
    python run_all.py

# Full dataset (paper results; set LIGHTWEIGHT_MODE=False in config.py first):
    python run_all.py --full

# Skip already-completed phases (resume after crash):
    python run_all.py --skip-to features
    python run_all.py --skip-to models_ml
    python run_all.py --skip-to models_dl
    python run_all.py --skip-to evaluation
    python run_all.py --skip-to robustness
    python run_all.py --skip-to final

Phase order and expected runtimes (lightweight / full HPC):
  A-C  pipeline_runner  Data + CUSUM + Labeling    ~30s  / ~10 min
  D    features         Feature engineering         ~15s  / ~5 min
  E    models_ml        LightGBM / XGBoost / RF     ~1 min / ~2 hrs
  F    models_dl        RNN / LSTM / GRU            ~30 min / ~10 hrs
  G    evaluation       Table 4/5 + Figures 4-6     ~5 min / ~1 hr
  H    robustness       Table 8 + Figure 10         ~5 min / ~30 min
  I    generate_remaining  Tables 1/2/6 + Figs 2/3/8 + Notebook  ~1 min
"""

import argparse
import gc
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.utils import setup_logging, set_reproducibility

logger = setup_logging(__name__)

PHASE_ORDER = ["pipeline", "features", "models_ml", "models_dl",
               "evaluation", "robustness", "final"]


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s // 60}m {s % 60}s"


def run_phase(name: str) -> None:
    """Run a single phase and log timing."""
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("STARTING PHASE: %s", name.upper())
    logger.info("=" * 60)

    if name == "pipeline":
        from src.pipeline_runner import run_phase_a, run_phase_b, run_phase_c
        run_phase_a(); run_phase_b(); run_phase_c()

    elif name == "features":
        # Phase 3: feature engineering
        runpy.run_module("src.features", run_name="__main__", alter_sys=True)

    elif name == "models_ml":
        # Phase 5: ML model training
        runpy.run_module("src.models_ml", run_name="__main__", alter_sys=True)

    elif name == "models_dl":
        # Phase 6: DL model training
        runpy.run_module("src.models_dl", run_name="__main__", alter_sys=True)

    elif name == "evaluation":
        # Phase 7: labeling + leakage experiments
        runpy.run_module("src.evaluation", run_name="__main__", alter_sys=True)

    elif name == "robustness":
        # Phase 9: robustness checks
        runpy.run_module("src.robustness", run_name="__main__", alter_sys=True)

    elif name == "final":
        # Phase 10: remaining tables, figures, notebook
        runpy.run_path(
            str(config.PROJECT_ROOT / "generate_remaining.py"),
            run_name="__main__",
        )

    gc.collect()
    logger.info("Phase %s finished in %s", name.upper(), _elapsed(t0))


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline runner")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Set LIGHTWEIGHT_MODE=False at runtime (override config.py)",
    )
    parser.add_argument(
        "--skip-to",
        metavar="PHASE",
        choices=PHASE_ORDER,
        default=None,
        help="Skip all phases before PHASE (resume from checkpoint)",
    )
    args = parser.parse_args()

    if args.full:
        config.LIGHTWEIGHT_MODE = False
        logger.info("FULL MODE: LIGHTWEIGHT_MODE overridden to False")
    else:
        logger.info(
            "LIGHTWEIGHT_MODE=%s (last %d rows)",
            config.LIGHTWEIGHT_MODE,
            config.SAMPLE_ROWS,
        )

    set_reproducibility(config.RANDOM_SEED)

    # Determine which phases to run
    start_idx = 0
    if args.skip_to:
        start_idx = PHASE_ORDER.index(args.skip_to)
        logger.info("Resuming from phase: %s (skipping %d phase(s))",
                    args.skip_to, start_idx)

    t_total = time.time()
    for phase in PHASE_ORDER[start_idx:]:
        run_phase(phase)

    logger.info("=" * 60)
    logger.info("ALL PHASES COMPLETE in %s", _elapsed(t_total))
    logger.info("Results: %s", config.RESULTS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
