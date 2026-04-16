"""
test_purged_cv.py — Unit tests for PurgedEmbargoKFold.
Master's Thesis: Leakage-Aware ML Framework for Flash Crash Detection

Tests verify the three leakage-prevention guarantees:
  1. No temporal overlap between training label horizons and test periods
  2. Embargo is Timedelta-based (not index-count-based)
  3. Embargo removes only AFTER the test fold, not before it
  4. Train sizes are reasonable (~50-85% of total events per fold)
  5. All n_splits folds yield non-empty train and test arrays
  6. Works correctly with irregularly spaced timestamps (weekend gaps)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.purged_cv import PurgedEmbargoKFold  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_dataset(
    n_events: int = 200,
    freq: str = "1min",
    horizon_bars: int = 10,
    start: str = "2024-01-01",
    tz: str = "UTC",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build a synthetic event dataset for testing.

    Parameters
    ----------
    n_events : int
        Number of events (rows).
    freq : str
        Pandas frequency string for the DatetimeIndex.
    horizon_bars : int
        t1 = event_time + horizon_bars * freq (fixed horizon).
    start : str
        Start datetime string.
    tz : str
        Timezone label for the DatetimeIndex.
    seed : int
        Random seed for label generation.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with DatetimeIndex.
    y : pd.Series
        Binary labels.
    t1 : pd.Series
        Label end-times (tz-naive to match real project data format).
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=n_events, freq=freq, tz=tz)
    X = pd.DataFrame(
        {"feat_a": rng.standard_normal(n_events).astype("float32"),
         "feat_b": rng.standard_normal(n_events).astype("float32")},
        index=index,
    )
    y = pd.Series(rng.integers(0, 2, size=n_events), index=index, name="label")

    # t1 is tz-naive (same as real project: parquet strips tz from t1 column)
    t1_vals = index + pd.Timedelta(freq) * horizon_bars
    t1 = pd.Series(
        t1_vals.tz_localize(None),   # strip tz to match real data format
        index=index,
        name="t1",
    )
    return X, y, t1


def make_irregular_dataset(seed: int = 42) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build a dataset with gaps (simulating FX market weekends/holidays).

    Creates two 80-event Monday-Friday blocks separated by a 2-day gap.
    """
    rng = np.random.default_rng(seed)
    # Block 1: Mon–Fri, 1-min bars, 80 events
    block1 = pd.date_range("2024-01-08 00:00", periods=80, freq="1min", tz="UTC")
    # Block 2: starts after a 2-day gap
    block2 = pd.date_range("2024-01-10 03:00", periods=80, freq="1min", tz="UTC")
    index = block1.append(block2)

    X = pd.DataFrame(
        {"feat_a": rng.standard_normal(len(index)).astype("float32")},
        index=index,
    )
    y = pd.Series(rng.integers(0, 2, size=len(index)), index=index, name="label")

    t1_vals = index + pd.Timedelta("5min")
    t1 = pd.Series(t1_vals.tz_localize(None), index=index, name="t1")
    return X, y, t1


# ---------------------------------------------------------------------------
# Test 1 — No temporal overlap after purging
# ---------------------------------------------------------------------------

class TestNoTemporalOverlap:
    """After purging, no training event's [event_time, t1] may overlap
    the test period [test_start, test_end]."""

    def test_no_temporal_overlap_regular(self) -> None:
        X, y, t1 = make_dataset(n_events=300, horizon_bars=15)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)

        t1_utc = pd.DatetimeIndex(t1.dt.tz_localize("UTC"))
        total_violations = 0

        for train_idx, test_idx in cv.split(X, y):
            test_start = X.index[test_idx[0]]
            test_end = X.index[test_idx[-1]]

            train_event_times = X.index[train_idx]
            train_t1_times = t1_utc[train_idx]

            # Overlap condition: t1_train > test_start AND event_train < test_end
            overlaps = (train_t1_times > test_start) & (train_event_times < test_end)
            total_violations += int(overlaps.sum())

        assert total_violations == 0, (
            f"Found {total_violations} training events whose label horizon "
            "overlaps the test period — purging is broken."
        )

    def test_no_temporal_overlap_with_long_horizon(self) -> None:
        """Longer horizons cross more fold boundaries — purging must handle all."""
        X, y, t1 = make_dataset(n_events=300, horizon_bars=50)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)

        t1_utc = pd.DatetimeIndex(t1.dt.tz_localize("UTC"))
        total_violations = 0

        for train_idx, test_idx in cv.split(X, y):
            test_start = X.index[test_idx[0]]
            test_end = X.index[test_idx[-1]]
            train_t1_times = t1_utc[train_idx]
            train_event_times = X.index[train_idx]

            overlaps = (train_t1_times > test_start) & (train_event_times < test_end)
            total_violations += int(overlaps.sum())

        assert total_violations == 0


# ---------------------------------------------------------------------------
# Test 2 — Embargo is time-based (Timedelta), not index-count-based
# ---------------------------------------------------------------------------

class TestEmbargoIsTimeBased:
    """The embargo zone must be a Timedelta, not an integer event count."""

    def test_embargo_duration_is_timedelta(self) -> None:
        X, y, t1 = make_dataset(n_events=200)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)
        info = cv.get_split_info(X)

        for fold, row in info.iterrows():
            assert isinstance(row["embargo_duration"], pd.Timedelta), (
                f"Fold {fold}: embargo_duration is {type(row['embargo_duration'])}, "
                "expected pd.Timedelta."
            )

    def test_embargo_proportional_to_time_span(self) -> None:
        """Doubling the dataset time span should double the embargo duration."""
        X_short, _, t1_short = make_dataset(n_events=200, freq="1min")
        X_long, _, t1_long = make_dataset(n_events=200, freq="2min")  # 2× time span

        cv_short = PurgedEmbargoKFold(n_splits=5, t1=t1_short, embargo_pct=0.01)
        cv_long = PurgedEmbargoKFold(n_splits=5, t1=t1_long, embargo_pct=0.01)

        info_short = cv_short.get_split_info(X_short)
        info_long = cv_long.get_split_info(X_long)

        emb_short = info_short["embargo_duration"].iloc[0]
        emb_long = info_long["embargo_duration"].iloc[0]

        # Long dataset has 2× the time span -> 2× the embargo
        ratio = emb_long / emb_short
        assert abs(ratio - 2.0) < 0.01, (
            f"Expected embargo to double with 2× time span, got ratio={ratio:.3f}. "
            "Embargo must be Timedelta-based, not index-count-based."
        )

    def test_embargo_end_is_after_test_end(self) -> None:
        """embargo_end must always be strictly after test_end."""
        X, y, t1 = make_dataset(n_events=200)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)
        info = cv.get_split_info(X)

        for fold, row in info.iterrows():
            assert row["embargo_end"] > row["test_end"], (
                f"Fold {fold}: embargo_end ({row['embargo_end']}) is not after "
                f"test_end ({row['test_end']})."
            )


# ---------------------------------------------------------------------------
# Test 3 — Embargo removes ONLY after test, not before
# ---------------------------------------------------------------------------

class TestEmbargoOnlyAfterTest:
    """Events before the test fold must not be removed by the embargo.
    They may only be removed by the purge step."""

    def test_no_before_test_events_embargoed(self) -> None:
        """Use zero-duration barriers so purging removes nothing.
        All pre-test events must then be in the training set."""
        # t1 = event_time (zero-duration barrier -> no purge overlap possible
        #   because t1_train == test_start is NOT strictly greater)
        n = 200
        X, y, _ = make_dataset(n_events=n)

        # t1 exactly equals event_time: interval is a point, no overlap with test
        t1_zero = pd.Series(
            X.index.tz_localize(None),   # same time, tz-naive
            index=X.index,
            name="t1",
        )
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1_zero, embargo_pct=0.01)

        fold_size = n // 5
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            t_start = fold_idx * fold_size
            # Integer positions strictly before the test block
            before_test_positions = set(range(0, t_start))
            train_set = set(train_idx.tolist())

            # All pre-test positions must be in train (embargo can't affect them)
            missing = before_test_positions - train_set
            assert len(missing) == 0, (
                f"Fold {fold_idx + 1}: {len(missing)} pre-test events are missing "
                "from training. Bug 2 (embargo before test) is not fixed."
            )

    def test_only_post_test_events_potentially_embargoed(self) -> None:
        """Every event that is removed AND lies before the test start must be
        due to purging only — verify via position check."""
        X, y, t1 = make_dataset(n_events=200, horizon_bars=10)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)

        fold_size = len(X) // 5
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            t_start = fold_idx * fold_size
            t_end = test_idx[-1] + 1 if fold_idx < 4 else len(X)

            # Identify all dropped events BEFORE the test block
            before_test_positions = set(range(0, t_start))
            train_before = set(train_idx[train_idx < t_start].tolist())
            dropped_before = before_test_positions - train_before

            # Each dropped-before event must satisfy the purge condition
            t1_utc = pd.DatetimeIndex(t1.dt.tz_localize("UTC"))
            test_start_time = X.index[t_start]
            test_end_time = X.index[test_idx[-1]]

            for pos in dropped_before:
                ev_time = X.index[pos]
                lbl_time = t1_utc[pos]
                is_purge = (lbl_time > test_start_time) and (ev_time < test_end_time)
                assert is_purge, (
                    f"Fold {fold_idx + 1}: event at position {pos} "
                    f"(time={ev_time}) is dropped from training but is "
                    "BEFORE the test fold and does NOT satisfy the purge "
                    "condition. This indicates the embargo is wrongly applied "
                    "before the test fold (Bug 2)."
                )


# ---------------------------------------------------------------------------
# Test 4 — Train size is reasonable per fold
# ---------------------------------------------------------------------------

class TestTrainSizeReasonable:
    """Each fold's training set should use 50–85% of total events.
    (80% baseline minus purged/embargoed events.)"""

    def test_train_pct_in_range(self) -> None:
        X, y, t1 = make_dataset(n_events=500, horizon_bars=10)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)
        n = len(X)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            train_pct = len(train_idx) / n * 100
            assert 50 <= train_pct <= 85, (
                f"Fold {fold_idx + 1}: train_pct={train_pct:.1f}% is outside "
                "expected range [50%, 85%]. Check purge/embargo logic."
            )

    def test_train_pct_from_get_split_info(self) -> None:
        X, y, t1 = make_dataset(n_events=500, horizon_bars=10)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)
        info = cv.get_split_info(X)

        for fold, row in info.iterrows():
            assert 50 <= row["train_pct"] <= 85, (
                f"Fold {fold}: train_pct={row['train_pct']}% out of range [50%, 85%]."
            )

    def test_split_and_get_split_info_agree(self) -> None:
        """split() and get_split_info() must report the same train sizes."""
        X, y, t1 = make_dataset(n_events=300, horizon_bars=10)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)
        info = cv.get_split_info(X)

        for fold_idx, (train_idx, _) in enumerate(cv.split(X, y)):
            expected_n_train = info.loc[fold_idx + 1, "n_train"]
            assert len(train_idx) == expected_n_train, (
                f"Fold {fold_idx + 1}: split() returned {len(train_idx)} train events "
                f"but get_split_info() reports {expected_n_train}."
            )


# ---------------------------------------------------------------------------
# Test 5 — All folds yield non-empty splits
# ---------------------------------------------------------------------------

class TestAllFoldsYield:
    """All n_splits folds must produce non-empty train and test arrays."""

    def test_all_folds_non_empty(self) -> None:
        X, y, t1 = make_dataset(n_events=200)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)
        folds_seen = 0

        for train_idx, test_idx in cv.split(X, y):
            assert len(train_idx) > 0, f"Fold {folds_seen + 1}: training set is empty."
            assert len(test_idx) > 0, f"Fold {folds_seen + 1}: test set is empty."
            folds_seen += 1

        assert folds_seen == 5, f"Expected 5 folds, got {folds_seen}."

    def test_correct_fold_count(self) -> None:
        for n_splits in [2, 3, 5, 10]:
            X, y, t1 = make_dataset(n_events=n_splits * 30)
            cv = PurgedEmbargoKFold(n_splits=n_splits, t1=t1, embargo_pct=0.01)
            count = sum(1 for _ in cv.split(X, y))
            assert count == n_splits, (
                f"n_splits={n_splits}: expected {n_splits} folds, got {count}."
            )

    def test_test_indices_cover_all_events(self) -> None:
        """Test indices across all folds must collectively cover every event."""
        n = 200
        X, y, t1 = make_dataset(n_events=n)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)

        all_test = np.concatenate([test for _, test in cv.split(X, y)])
        assert len(all_test) == n, (
            f"Test indices across folds cover {len(all_test)} events, "
            f"expected {n}."
        )
        assert len(np.unique(all_test)) == n, (
            "Some events appear in multiple test folds (test fold overlap)."
        )

    def test_no_train_test_overlap_within_fold(self) -> None:
        """Train and test index sets must be disjoint within each fold."""
        X, y, t1 = make_dataset(n_events=200)
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, (
                f"Fold {fold_idx + 1}: {len(overlap)} indices appear in both "
                "train and test sets."
            )


# ---------------------------------------------------------------------------
# Test 6 — Works with irregular spacing (weekend gaps)
# ---------------------------------------------------------------------------

class TestIrregularSpacing:
    """The CV must work correctly when timestamps are not uniformly spaced."""

    def test_irregular_spacing_no_overlap(self) -> None:
        """No purge violation on dataset with a large mid-series gap."""
        X, y, t1 = make_irregular_dataset()
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)

        t1_utc = pd.DatetimeIndex(t1.dt.tz_localize("UTC"))
        total_violations = 0

        for train_idx, test_idx in cv.split(X, y):
            test_start = X.index[test_idx[0]]
            test_end = X.index[test_idx[-1]]
            train_event_times = X.index[train_idx]
            train_t1_times = t1_utc[train_idx]

            overlaps = (train_t1_times > test_start) & (train_event_times < test_end)
            total_violations += int(overlaps.sum())

        assert total_violations == 0, (
            f"Irregular dataset: {total_violations} purge violations found."
        )

    def test_irregular_spacing_embargo_spans_gap(self) -> None:
        """When the test fold ends just before a gap, the embargo must span
        the gap correctly (time-based, not event-count-based)."""
        X, y, t1 = make_irregular_dataset()
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.05)
        info = cv.get_split_info(X)

        # All embargo durations must be pd.Timedelta
        for fold, row in info.iterrows():
            assert isinstance(row["embargo_duration"], pd.Timedelta), (
                f"Fold {fold}: embargo_duration is not a Timedelta."
            )
            assert row["embargo_end"] > row["test_end"], (
                f"Fold {fold}: embargo_end not after test_end."
            )

    def test_all_folds_yield_on_irregular(self) -> None:
        X, y, t1 = make_irregular_dataset()
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=0.01)
        folds = list(cv.split(X, y))
        assert len(folds) == 5
        for i, (train_idx, test_idx) in enumerate(folds):
            assert len(train_idx) > 0, f"Fold {i + 1}: empty training set on irregular data."
            assert len(test_idx) > 0, f"Fold {i + 1}: empty test set on irregular data."


# ---------------------------------------------------------------------------
# Test 7 — Validation guards
# ---------------------------------------------------------------------------

class TestValidation:
    """Constructor and split() should raise on invalid inputs."""

    def test_n_splits_lt_2_raises(self) -> None:
        _, _, t1 = make_dataset(n_events=50)
        with pytest.raises(ValueError, match="n_splits"):
            PurgedEmbargoKFold(n_splits=1, t1=t1)

    def test_embargo_pct_out_of_range_raises(self) -> None:
        _, _, t1 = make_dataset(n_events=50)
        with pytest.raises(ValueError, match="embargo_pct"):
            PurgedEmbargoKFold(n_splits=5, t1=t1, embargo_pct=1.5)

    def test_unsorted_index_raises(self) -> None:
        X, y, t1 = make_dataset(n_events=100)
        X_shuffled = X.sample(frac=1, random_state=0)  # shuffle rows
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1)
        with pytest.raises(ValueError, match="sorted"):
            next(cv.split(X_shuffled, y))

    def test_t1_length_mismatch_raises(self) -> None:
        X, y, t1 = make_dataset(n_events=100)
        t1_short = t1.iloc[:50]
        cv = PurgedEmbargoKFold(n_splits=5, t1=t1_short)
        with pytest.raises(ValueError, match="t1 length"):
            next(cv.split(X, y))
