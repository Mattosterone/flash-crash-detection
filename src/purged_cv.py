"""
purged_cv.py — Module 6: Purged-Embargo Cross-Validation.
Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection

Implements walk-forward K-Fold CV with three leakage safeguards:
1. TEMPORAL ORDER  — test folds are contiguous future time blocks
2. PURGING         — removes training events whose label horizon [t, t1]
                     overlaps ANY part of the test period
3. EMBARGO         — removes a TIME-BASED buffer zone AFTER the test fold

Fixes three bugs present in legacy code (documented in CLAUDE.md):
  Bug 1 (index-based embargo)  → embargo_td = total_span * embargo_pct
  Bug 2 (remove before test)   → embargo removes only AFTER test end
  Bug 3 (incomplete purge)     → full overlap check: t1 > test_start AND
                                  event_time < test_end
"""

import logging
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

import config
from src.utils import setup_logging

logger = setup_logging(__name__)


class PurgedEmbargoKFold:
    """Walk-forward K-Fold CV with purging and time-based embargo.

    Splits a time-indexed event dataset into K contiguous folds and
    ensures no label-horizon leakage between training and test sets.

    Parameters
    ----------
    n_splits : int
        Number of CV folds.  Must be >= 2.
    t1 : pd.Series
        Label end-times indexed by event datetime.  Each value is the
        timestamp when the triple barrier was first hit.  Must be
        aligned with the ``X`` passed to :meth:`split`.
    embargo_pct : float
        Embargo as a fraction of the total time span of the dataset.
        Default: ``config.CV_PCT_EMBARGO``.

    Notes
    -----
    The full overlap purge condition is::

        purge if  t1_train > test_start  AND  event_train < test_end

    This is stricter than checking only the label endpoint and prevents
    any interval overlap between [event_train, t1_train] and
    [test_start, test_end].
    """

    def __init__(
        self,
        n_splits: int,
        t1: pd.Series,
        embargo_pct: float = config.CV_PCT_EMBARGO,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not (0.0 <= embargo_pct < 1.0):
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")

        self.n_splits = n_splits
        self.t1 = t1.copy()
        self.embargo_pct = embargo_pct

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_t1_timezone(self, X: pd.DataFrame) -> pd.Series:
        """Return a copy of self.t1 with timezone matching X.index.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix whose DatetimeIndex defines the reference tz.

        Returns
        -------
        pd.Series
            t1 values with timezone consistent with X.index.
        """
        t1 = self.t1.copy()
        ref_tz = X.index.tz

        try:
            t1_tz = t1.dt.tz
        except AttributeError:
            t1_tz = None

        if ref_tz is not None and t1_tz is None:
            # X.index is tz-aware, t1 is tz-naive → localize t1 to same tz
            t1 = t1.dt.tz_localize(ref_tz)
        elif ref_tz is None and t1_tz is not None:
            # X.index is tz-naive, t1 is tz-aware → strip tz from t1
            t1 = t1.dt.tz_localize(None)
        # If both tz-aware with different zones, convert t1 to ref_tz
        elif ref_tz is not None and t1_tz is not None and ref_tz != t1_tz:
            t1 = t1.dt.tz_convert(ref_tz)

        return t1

    def _compute_fold_bounds(self, n: int) -> list[tuple[int, int]]:
        """Return (start_pos, end_pos) pairs for each fold.

        Distributes remainder events into the last fold so all folds
        have at least n // n_splits events.

        Parameters
        ----------
        n : int
            Total number of events.

        Returns
        -------
        list of (int, int)
            Inclusive-start, exclusive-end integer position pairs.
        """
        fold_size = n // self.n_splits
        bounds = []
        for k in range(self.n_splits):
            start = k * fold_size
            end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            bounds.append((start, end))
        return bounds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test integer index arrays for each fold.

        Each fold uses a future contiguous block as test and all remaining
        events (minus purged + embargoed) as training.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with a monotonically increasing DatetimeIndex.
        y : pd.Series, optional
            Labels — ignored; present for sklearn API compatibility.
        groups : pd.Series, optional
            Not used; present for sklearn API compatibility.

        Yields
        ------
        train_idx : np.ndarray
            Integer positions of retained training events.
        test_idx : np.ndarray
            Integer positions of test events.

        Raises
        ------
        ValueError
            If X.index is not sorted or t1 length mismatches X.
        """
        if not X.index.is_monotonic_increasing:
            raise ValueError(
                "X.index must be sorted in ascending order. "
                "Call X.sort_index() before passing to split()."
            )
        n = len(X)
        if n < self.n_splits * 2:
            raise ValueError(
                f"Dataset has only {n} events — too small for {self.n_splits} splits."
            )
        if len(self.t1) != n:
            raise ValueError(
                f"t1 length ({len(self.t1)}) must equal X length ({n})."
            )

        # Bug 1 fix: embargo duration as Timedelta, not index count
        total_span: pd.Timedelta = X.index[-1] - X.index[0]
        embargo_td: pd.Timedelta = total_span * self.embargo_pct
        logger.debug(
            "PurgedEmbargoKFold | n=%d, n_splits=%d, embargo_pct=%.4f, "
            "total_span=%s, embargo_td=%s",
            n, self.n_splits, self.embargo_pct, total_span, embargo_td,
        )

        t1_aligned = self._align_t1_timezone(X)
        fold_bounds = self._compute_fold_bounds(n)

        # Pre-compute int64 nanosecond arrays for fast vectorised comparison
        # asi8 on a DatetimeIndex/DatetimeTZDtype always returns UTC nanoseconds
        index_ns: np.ndarray = X.index.asi8                          # shape (n,)
        t1_ns: np.ndarray = pd.DatetimeIndex(t1_aligned).asi8        # shape (n,)

        for fold_idx, (t_start, t_end) in enumerate(fold_bounds):
            test_idx = np.arange(t_start, t_end)

            # Scalar boundaries in nanoseconds
            test_start_ns: int = index_ns[t_start]
            test_end_ns: int = index_ns[t_end - 1]
            embargo_end_ns: int = test_end_ns + int(embargo_td.total_seconds() * 1e9)

            # Training candidates: everything outside the test block
            train_candidates = np.concatenate(
                [np.arange(0, t_start), np.arange(t_end, n)]
            )

            ev_ns = index_ns[train_candidates]      # event times (ns)
            lbl_ns = t1_ns[train_candidates]        # label end times (ns)

            # Bug 3 fix: full interval overlap check
            #   purge if [event, t1] ∩ [test_start, test_end] ≠ ∅
            #   ⟺  t1_train > test_start  AND  event_train < test_end
            purge_mask: np.ndarray = (lbl_ns > test_start_ns) & (ev_ns < test_end_ns)

            # Bug 2 fix: embargo only AFTER test fold, never before
            embargo_mask: np.ndarray = (ev_ns > test_end_ns) & (ev_ns <= embargo_end_ns)

            keep_mask = ~purge_mask & ~embargo_mask
            final_train = train_candidates[keep_mask]

            logger.info(
                "Fold %d/%d | test=[%s → %s] | "
                "train=%d  purged=%d  embargoed=%d",
                fold_idx + 1, self.n_splits,
                pd.Timestamp(test_start_ns, unit="ns", tz=X.index.tz).date(),
                pd.Timestamp(test_end_ns, unit="ns", tz=X.index.tz).date(),
                len(final_train),
                int(purge_mask.sum()),
                int(embargo_mask.sum()),
            )

            yield final_train, test_idx

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> int:
        """Return the number of splits (required by sklearn CV interface).

        Parameters
        ----------
        X, y, groups : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        int
            ``self.n_splits``
        """
        return self.n_splits

    def get_split_info(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a per-fold summary DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with a monotonically increasing DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            One row per fold with columns:
            ``test_start``, ``test_end``, ``embargo_end``,
            ``n_train``, ``n_test``, ``n_purged``, ``n_embargoed``,
            ``train_pct``, ``test_pct``, ``embargo_duration``.
            Index is the fold number (1-based).
        """
        if not X.index.is_monotonic_increasing:
            raise ValueError("X.index must be sorted in ascending order.")
        if len(self.t1) != len(X):
            raise ValueError(
                f"t1 length ({len(self.t1)}) must equal X length ({len(X)})."
            )

        n = len(X)
        total_span: pd.Timedelta = X.index[-1] - X.index[0]
        embargo_td: pd.Timedelta = total_span * self.embargo_pct
        tz = X.index.tz

        t1_aligned = self._align_t1_timezone(X)
        fold_bounds = self._compute_fold_bounds(n)

        index_ns: np.ndarray = X.index.asi8
        t1_ns: np.ndarray = pd.DatetimeIndex(t1_aligned).asi8

        rows = []
        for fold_idx, (t_start, t_end) in enumerate(fold_bounds):
            test_start_ns = index_ns[t_start]
            test_end_ns = index_ns[t_end - 1]
            embargo_end_ns = test_end_ns + int(embargo_td.total_seconds() * 1e9)

            train_candidates = np.concatenate(
                [np.arange(0, t_start), np.arange(t_end, n)]
            )
            ev_ns = index_ns[train_candidates]
            lbl_ns = t1_ns[train_candidates]

            purge_mask = (lbl_ns > test_start_ns) & (ev_ns < test_end_ns)
            embargo_mask = (ev_ns > test_end_ns) & (ev_ns <= embargo_end_ns)
            keep_mask = ~purge_mask & ~embargo_mask

            n_train = int(keep_mask.sum())
            n_test = t_end - t_start
            n_purged = int(purge_mask.sum())
            n_embargoed = int(embargo_mask.sum())

            rows.append({
                "fold": fold_idx + 1,
                "test_start": pd.Timestamp(test_start_ns, unit="ns", tz=tz),
                "test_end": pd.Timestamp(test_end_ns, unit="ns", tz=tz),
                "embargo_end": pd.Timestamp(embargo_end_ns, unit="ns", tz=tz),
                "n_train": n_train,
                "n_test": n_test,
                "n_purged": n_purged,
                "n_embargoed": n_embargoed,
                "n_total": n,
                "train_pct": round(n_train / n * 100, 1),
                "test_pct": round(n_test / n * 100, 1),
                "embargo_duration": embargo_td,
            })

        return pd.DataFrame(rows).set_index("fold")
