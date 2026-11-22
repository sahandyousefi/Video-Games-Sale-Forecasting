from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vgsales.config import config_from_dict, load_config
from vgsales.data.cleaning import clean
from vgsales.data.schema import LEAKAGE_COLUMNS, TARGET
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.dataset import Dataset, load_dataset, prepare_dataset
from vgsales.features.prerelease import (
    GROUP_COLUMN,
    OUTPUT_COLUMNS,
    build_prerelease_features,
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return make_synthetic(SyntheticSpec(n_titles=500), seed=12)


@pytest.fixture(scope="module")
def dataset(raw: pd.DataFrame) -> Dataset:
    return prepare_dataset(raw, load_config())


def test_features_and_target_are_aligned(raw: pd.DataFrame, dataset: Dataset) -> None:
    scoped, _ = apply_release_scope(clean(raw)[0], load_config().data)
    assert list(dataset.features.columns) == list(OUTPUT_COLUMNS)
    assert not set(LEAKAGE_COLUMNS) & set(dataset.features.columns)
    np.testing.assert_allclose(dataset.target, scoped[TARGET].to_numpy())
    np.testing.assert_array_equal(dataset.censored, scoped["censored"].to_numpy())
    assert (dataset.target >= 0).all()


def test_holdout_takes_the_latest_years_and_purges_titles(dataset: Dataset) -> None:
    years = dataset.features["release_year"].to_numpy()
    holdout_years = load_config().split.holdout_years
    assert years[dataset.test_idx].min() == years.max() - holdout_years + 1
    assert years[dataset.dev_idx].max() < years[dataset.test_idx].min()
    dev_titles = set(dataset.groups.iloc[dataset.dev_idx])
    assert not dev_titles & set(dataset.groups.iloc[dataset.test_idx])


def test_development_and_holdout_accessors(dataset: Dataset) -> None:
    X_dev, y_dev = dataset.development()
    X_test, y_test = dataset.holdout()
    assert len(X_dev) == y_dev.size == dataset.dev_idx.size
    assert len(X_test) == y_test.size == dataset.test_idx.size
    assert X_dev.index.intersection(X_test.index).empty


def test_summary_counts(dataset: Dataset) -> None:
    summary = dataset.summary()
    assert summary["n_rows"] == len(dataset.features)
    total = (
        summary["development"]["n_rows"]
        + summary["holdout"]["n_rows"]
        + summary["n_purged"]
    )
    assert total == summary["n_rows"]
    assert summary["development"]["last_year"] < summary["holdout"]["first_year"]
    assert summary["holdout"]["n_censored"] >= summary["development"]["n_censored"]


def test_untracked_releases_inform_platform_features(
    raw: pd.DataFrame, dataset: Dataset
) -> None:
    scoped, _ = apply_release_scope(clean(raw)[0], load_config().data)
    tracked_only = build_prerelease_features(scoped)
    # A larger reference can only add earlier platforms and earlier launches.
    assert (dataset.features["n_platforms"] >= tracked_only["n_platforms"]).all()
    assert (dataset.features["n_platforms"] > tracked_only["n_platforms"]).any()
    age, age_tracked = (
        dataset.features["platform_age_years"],
        tracked_only["platform_age_years"],
    )
    assert (age.fillna(0) >= age_tracked.fillna(0) - 1e-12).all()


def test_holdout_length_follows_the_configuration(raw: pd.DataFrame) -> None:
    config = config_from_dict({"split": {"holdout_years": 1}})
    one_year = prepare_dataset(raw, config)
    years = one_year.features["release_year"].to_numpy()
    assert set(years[one_year.test_idx]) == {years.max()}


def test_empty_scope_raises(raw: pd.DataFrame) -> None:
    config = config_from_dict(
        {"data": {"min_release_year": 1980, "max_release_year": 1981}}
    )
    with pytest.raises(ValueError, match="no rows"):
        prepare_dataset(raw, config)


def test_load_dataset_reads_csv(raw: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "extract.csv"
    raw.to_csv(path, index=False)
    from_disk = load_dataset(load_config(), path)
    in_memory = prepare_dataset(raw, load_config())
    pd.testing.assert_frame_equal(from_disk.features, in_memory.features)
    np.testing.assert_allclose(from_disk.target, in_memory.target)
    with pytest.raises(FileNotFoundError):
        load_dataset(load_config(), tmp_path / "missing.csv")


def test_group_column_is_kept_for_splitting(dataset: Dataset) -> None:
    assert dataset.groups.name == GROUP_COLUMN
    assert dataset.groups.notna().all()
