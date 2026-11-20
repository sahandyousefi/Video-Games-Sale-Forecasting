from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score

from vgsales.config import SplitConfig, load_config
from vgsales.data.cleaning import clean
from vgsales.data.grouping import title_key
from vgsales.data.splits import (
    ExpandingWindowSplit,
    describe_folds,
    expanding_window_folds,
    holdout_split,
)
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope


@pytest.fixture(scope="module")
def years() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(1995, 2021, size=600)


def test_holdout_takes_latest_calendar_years(years: np.ndarray) -> None:
    dev, test = holdout_split(years, holdout_years=3)
    assert set(years[test]) == {2018, 2019, 2020}
    assert years[dev].max() == 2017
    assert np.intersect1d(dev, test).size == 0
    assert dev.size + test.size == years.size


def test_holdout_accepts_series(years: np.ndarray) -> None:
    series = pd.Series(years, index=np.arange(years.size) + 1000)
    dev, test = holdout_split(series, holdout_years=2)
    assert series.iloc[test].min() == 2019
    assert dev.max() < years.size


@pytest.mark.parametrize("holdout_years", [0, 30])
def test_invalid_holdout_raises(years: np.ndarray, holdout_years: int) -> None:
    with pytest.raises(ValueError):
        holdout_split(years, holdout_years)


def test_missing_years_raise() -> None:
    with pytest.raises(ValueError, match="missing"):
        holdout_split(np.array([2000.0, np.nan]), 1)


def test_folds_validate_on_latest_years(years: np.ndarray) -> None:
    folds = expanding_window_folds(years, n_splits=4, min_train_years=5)
    assert [f.val_years for f in folds] == [
        (2017, 2017),
        (2018, 2018),
        (2019, 2019),
        (2020, 2020),
    ]


def test_training_precedes_validation(years: np.ndarray) -> None:
    for fold in expanding_window_folds(years, n_splits=5, min_train_years=5):
        assert years[fold.train_idx].max() < years[fold.val_idx].min()
        assert np.intersect1d(fold.train_idx, fold.val_idx).size == 0


def test_training_windows_expand(years: np.ndarray) -> None:
    folds = expanding_window_folds(years, n_splits=5, min_train_years=5)
    for earlier, later in pairwise(folds):
        assert set(earlier.train_idx) < set(later.train_idx)
        assert set(earlier.val_idx) <= set(later.train_idx)


def test_gap_years_are_excluded(years: np.ndarray) -> None:
    folds = expanding_window_folds(years, n_splits=3, min_train_years=5, gap_years=2)
    for fold in folds:
        assert years[fold.train_idx].max() == fold.val_years[0] - 3


def test_multi_year_validation_blocks(years: np.ndarray) -> None:
    folds = expanding_window_folds(years, n_splits=2, val_years=3)
    assert [f.val_years for f in folds] == [(2015, 2017), (2018, 2020)]
    assert set(years[folds[0].val_idx]) == {2015, 2016, 2017}


def test_too_few_training_years_raise(years: np.ndarray) -> None:
    with pytest.raises(ValueError, match="min_train_years"):
        expanding_window_folds(years, n_splits=5, min_train_years=25)


def test_empty_validation_block_raises() -> None:
    years = np.array([2000, 2001, 2002, 2004])
    with pytest.raises(ValueError, match="2003"):
        expanding_window_folds(years, n_splits=2)


def test_describe_folds(years: np.ndarray) -> None:
    folds = expanding_window_folds(years, n_splits=3, min_train_years=5)
    table = describe_folds(folds)
    assert table["fold"].tolist() == [0, 1, 2]
    assert table["train_start"].eq(1995).all()
    assert table["train_end"].tolist() == [2017, 2018, 2019]
    assert table["n_val"].sum() == sum(f.val_idx.size for f in folds)


def test_splitter_works_with_scikit_learn(years: np.ndarray) -> None:
    X = pd.DataFrame({"release_year": years, "x": np.arange(years.size)})
    y = np.log1p(np.arange(years.size) % 7)
    cv = ExpandingWindowSplit(n_splits=4, min_train_years=5)
    scores = cross_val_score(DummyRegressor(), X, y, cv=cv)
    assert scores.shape == (4,)
    assert cv.get_n_splits() == 4


def test_splitter_from_config() -> None:
    config = SplitConfig(cv_folds=3, min_train_years=7, gap_years=1)
    cv = ExpandingWindowSplit.from_config(config)
    assert (cv.n_splits, cv.min_train_years, cv.gap_years) == (3, 7, 1)
    assert "n_splits=3" in repr(cv)


def test_splitter_requires_time_column() -> None:
    cv = ExpandingWindowSplit(n_splits=2, min_train_years=1)
    with pytest.raises(ValueError, match="release_year"):
        list(cv.split(pd.DataFrame({"x": [1, 2, 3]})))
    with pytest.raises(ValueError, match="release_year"):
        list(cv.split(np.zeros((3, 2))))


def test_default_configuration_on_synthetic_data() -> None:
    config = load_config()
    raw = make_synthetic(SyntheticSpec(n_titles=600), seed=2)
    frame, _ = apply_release_scope(clean(raw)[0], config.data)
    dev_idx, test_idx = holdout_split(frame["release_year"], config.split.holdout_years)
    dev = frame.iloc[dev_idx].reset_index(drop=True)
    cv = ExpandingWindowSplit.from_config(config.split)
    table = describe_folds(cv.folds(dev))
    assert len(table) == config.split.cv_folds
    assert table["val_end"].max() == dev["release_year"].max()
    assert frame.iloc[test_idx]["release_year"].min() > table["val_end"].max()


def _ported_titles() -> tuple[np.ndarray, np.ndarray]:
    # Title "b" is ported two years later; "d" spans the holdout boundary.
    years = np.array([2010, 2011, 2012, 2013, 2013, 2014, 2015, 2016, 2016, 2017])
    groups = np.array(["a", "b", "c", "b", "e", "d", "f", "g", "d", "h"])
    return years, groups


def test_holdout_purges_groups_that_reach_the_test_years() -> None:
    years, groups = _ported_titles()
    dev, test = holdout_split(years, holdout_years=2, groups=groups)
    assert set(groups[test]) == {"g", "d", "h"}
    assert "d" not in set(groups[dev])
    assert set(dev) == set(np.flatnonzero(years <= 2015)) - {5}


def test_folds_keep_groups_on_one_side() -> None:
    years, groups = _ported_titles()
    folds = expanding_window_folds(years, n_splits=5, groups=groups)
    for fold in folds:
        assert not set(groups[fold.train_idx]) & set(groups[fold.val_idx])
        assert years[fold.train_idx].max() < years[fold.val_idx].min()
    by_year = {fold.val_years[0]: fold for fold in folds}
    # The 2011 release of "b" is purged when its 2013 port is validated.
    assert by_year[2013].n_purged == 1
    assert 1 not in by_year[2013].train_idx
    assert by_year[2014].n_purged == 0
    assert by_year[2016].n_purged == 1


def test_groups_on_synthetic_titles() -> None:
    frame, _ = apply_release_scope(
        clean(make_synthetic(SyntheticSpec(n_titles=400), seed=4))[0],
        load_config().data,
    )
    keys = title_key(frame["title"]).to_numpy()
    for fold in expanding_window_folds(
        frame["release_year"], n_splits=4, min_train_years=5, groups=keys
    ):
        assert not set(keys[fold.train_idx]) & set(keys[fold.val_idx])


def test_splitter_reads_groups_from_argument_or_column() -> None:
    years, groups = _ported_titles()
    X = pd.DataFrame({"release_year": years, "title_key": groups})
    from_arg = list(ExpandingWindowSplit(5, 1).split(X, groups=groups))
    from_col = list(ExpandingWindowSplit(5, 1, group_column="title_key").split(X))
    for (a_train, a_val), (b_train, b_val) in zip(from_arg, from_col, strict=True):
        np.testing.assert_array_equal(a_train, b_train)
        np.testing.assert_array_equal(a_val, b_val)
    y = np.arange(years.size, dtype=float)
    scores = cross_val_score(
        DummyRegressor(),
        X,
        y,
        cv=ExpandingWindowSplit(3, 1),
        groups=groups,
        scoring="neg_mean_absolute_error",
    )
    assert scores.shape == (3,)


def test_misaligned_or_missing_groups_raise() -> None:
    years, groups = _ported_titles()
    with pytest.raises(ValueError, match="length"):
        holdout_split(years, 1, groups=groups[:-1])
    with pytest.raises(ValueError, match="missing"):
        expanding_window_folds(years, 2, groups=np.array([None] * years.size))
