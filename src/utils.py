"""
utils.py — Module 12: Logging, timing, seeds, and I/O helpers.

Provides:
    set_reproducibility  : set all random seeds (numpy, random, torch)
    setup_logging        : configure file + console logging
    timer                : decorator that measures wall-clock execution time
    save_result          : persist a DataFrame/Series to parquet
    load_result          : load a parquet file back to DataFrame
"""

import functools
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

import config


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_reproducibility(seed: int = config.RANDOM_SEED) -> None:
    """Set random seeds for numpy, Python's random module, and PyTorch.

    Parameters
    ----------
    seed : int
        Seed value.  Defaults to config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed; skip silently
    logging.getLogger(__name__).debug("Reproducibility seeds set to %d", seed)


# ======================================================================
# LOGGING
# ======================================================================

def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Configure a logger with both file and console handlers.

    Creates one rotating log file per named logger under config.LOGS_DIR
    (or a supplied override).  Console handler uses a compact format;
    file handler uses the full timestamped format.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    level : int
        Logging level for both handlers (default: logging.INFO).
    log_dir : Path, optional
        Directory for the log file.  Defaults to config.LOGS_DIR.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if log_dir is None:
        log_dir = config.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt_full = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_short = logging.Formatter("%(levelname)-8s %(name)s  %(message)s")

    # File handler
    log_file = log_dir / f"{name.replace('.', '_')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt_full)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt_short)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ======================================================================
# TIMER DECORATOR
# ======================================================================

def timer(func: Callable) -> Callable:
    """Decorator that logs wall-clock execution time of a function.

    Logs at INFO level: "<module>.<func_name> completed in X.XXs".

    Parameters
    ----------
    func : Callable
        The function to wrap.

    Returns
    -------
    Callable
        Wrapped function with timing.

    Example
    -------
    >>> @timer
    ... def slow_fn():
    ...     time.sleep(1)
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2fs", func.__qualname__, elapsed)
        return result
    return wrapper


# ======================================================================
# RESULT I/O
# ======================================================================

def save_result(
    df: Union[pd.DataFrame, pd.Series],
    name: str,
    subdir: Optional[Path] = None,
) -> Path:
    """Persist a DataFrame or Series to parquet under the processed data dir.

    Parquet preserves DatetimeIndex and all dtypes, making it the
    canonical intermediate format for this project.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Data to save.  Series is converted to a single-column DataFrame.
    name : str
        File stem (without extension).  E.g. ``"events"`` -> ``events.parquet``.
    subdir : Path, optional
        Sub-directory inside config.PROCESSED_DATA_DIR.
        If None, saves directly to PROCESSED_DATA_DIR.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    base = config.PROCESSED_DATA_DIR if subdir is None else subdir
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}.parquet"

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df.to_parquet(path, index=True)
    logger = logging.getLogger(__name__)
    logger.info("Saved %s -> %s  (shape=%s)", name, path, df.shape)
    return path


def load_result(
    name: str,
    subdir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load a parquet file from the processed data dir.

    Parameters
    ----------
    name : str
        File stem (without extension).  E.g. ``"events"`` -> ``events.parquet``.
    subdir : Path, optional
        Sub-directory inside config.PROCESSED_DATA_DIR.
        If None, loads directly from PROCESSED_DATA_DIR.

    Returns
    -------
    pd.DataFrame
        Loaded data with original index and dtypes restored.

    Raises
    ------
    FileNotFoundError
        If the expected parquet file does not exist.
    """
    base = config.PROCESSED_DATA_DIR if subdir is None else subdir
    path = base / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}\n"
            f"Run the upstream pipeline step to generate it."
        )

    df = pd.read_parquet(path)
    logger = logging.getLogger(__name__)
    logger.info("Loaded %s ← %s  (shape=%s)", name, path, df.shape)
    return df
