import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from vgsales.config import FeatureConfig, config_from_dict, load_config
from vgsales.data.cleaning import clean
from vgsales.data.schema import LEAKAGE_COLUMNS, REGIONAL_SALES, TARGET
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.features.pipeline import UNKNOWN_CODE, LeakageGuard, make_preprocessor
from vgsales.features.prerelease import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    build_prerelease_features,
)


@pytest.fixture(scope="module")
def scoped() -> pd.DataFrame:
    raw = make_synthetic(SyntheticSpec(n_titles=600), seed=21)
    frame, _ = apply_release_scope(clean(raw)[0], load_config().data)
    return frame


@pytest.fixture(scope="module")
def data(scoped: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return build_prerelease_features(scoped), np.log1p(scoped[TARGET].to_numpy())


def _split(X: pd.DataFrame, last_train_year: int) -> tuple[np.ndarray, np.ndarray]:
    years = X["release_year"].to_numpy()
    return np.flatnonzero(years <= last_train_year), np.flatnonzero(
        years > last_train_year
    )


def test_default_config_columns_come_from_the_feature_builder() -> None:
    config = load_config().features
    configured = {*config.categorical, *config.high_cardinality, *config.numeric}
    assert configured <= set(FEATURE_COLUMNS)


def test_output_is_finite_and_named(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = data
    pre = make_preprocessor(load_config().features).set_output(transform="pandas")
    out = pre.fit_transform(X, y)
    assert len(out) == len(X)
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    names = set(out.columns)
    encoded = {"publisher_freq", "developer_freq", "publisher_te", "developer_te"}
    assert encoded <= names
    assert {"critic_score", "critic_missing", "release_year"} <= names
    assert any(name.startswith("console_") for name in names)
    assert GROUP_COLUMN not in names
    assert not names & {TARGET, *LEAKAGE_COLUMNS}


def test_regional_columns_are_rejected(
    scoped: pd.DataFrame, data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    X, y = data
    pre = make_preprocessor()
    for col in (*REGIONAL_SALES, TARGET, "last_update"):
        with pytest.raises(ValueError, match="leakage"):
            clone(pre).fit(X.assign(**{col: 1.0}), y)
    with pytest.raises(ValueError, match="leakage"):
        clone(pre).fit(scoped, y)
    fitted = clone(pre).fit(X, y)
    with pytest.raises(ValueError, match="leakage"):
        fitted.transform(X.assign(na_sales=0.5))


@pytest.mark.parametrize("field", ["categorical", "high_cardinality", "numeric"])
def test_configuring_a_leakage_column_fails(field: str) -> None:
    with pytest.raises(ValueError, match="leakage"):
        config_from_dict({"features": {field: ["pal_sales"]}})


def test_overlapping_column_groups_fail() -> None:
    with pytest.raises(ValueError, match="high_cardinality"):
        FeatureConfig(categorical=("publisher",))


def test_transform_is_row_wise(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = data
    train, val = _split(X, 2015)
    pre = make_preprocessor().fit(X.iloc[train], y[train])
    full = pre.transform(X.iloc[val])
    half = val.size // 2
    parts = np.vstack(
        [pre.transform(X.iloc[val[:half]]), pre.transform(X.iloc[val[half:]])]
    )
    np.testing.assert_allclose(parts, full)
    reordered = pre.transform(X.iloc[val[::-1]])
    np.testing.assert_allclose(reordered[::-1], full)


def test_future_rows_do_not_change_past_features(
    scoped: pd.DataFrame,
) -> None:
    # Features of the 2014-2015 block must be identical whether or not the
    # releases after 2015 exist in the data.
    last_val_year = 2015
    truncated = scoped[scoped["release_year"] <= last_val_year]
    results = []
    for frame in (scoped, truncated):
        X = build_prerelease_features(frame)
        y = np.log1p(frame[TARGET].to_numpy())
        years = X["release_year"].to_numpy()
        train = years <= last_val_year - 2
        val = (years > last_val_year - 2) & (years <= last_val_year)
        pre = make_preprocessor(scale=True).fit(X[train], y[train])
        results.append(pd.DataFrame(pre.transform(X[val]), index=X.index[val]))
    pd.testing.assert_frame_equal(results[0], results[1])


def test_validation_targets_do_not_reach_features(
    scoped: pd.DataFrame,
) -> None:
    X = build_prerelease_features(scoped)
    y = np.log1p(scoped[TARGET].to_numpy())
    train, val = _split(X, 2014)
    shuffled = y.copy()
    shuffled[val] = np.random.default_rng(1).permutation(y[val])
    a = make_preprocessor().fit(X.iloc[train], y[train]).transform(X)
    b = make_preprocessor().fit(X.iloc[train], shuffled[train]).transform(X)
    np.testing.assert_allclose(a, b)


def test_cross_validation_with_title_groups(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = data
    cv = ExpandingWindowSplit(n_splits=3, min_train_years=5, group_column=GROUP_COLUMN)
    model = make_pipeline(make_preprocessor(scale=True), Ridge(alpha=1.0))
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_pipeline_survives_pickling(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = data
    pre = make_preprocessor().fit(X, y)
    restored = pickle.loads(pickle.dumps(pre))
    np.testing.assert_allclose(restored.transform(X), pre.transform(X))


def test_ordinal_encoding_codes_pools_and_flags_unknown(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = data
    config = FeatureConfig(min_category_count=10)
    pre = make_preprocessor(config, categorical="ordinal")
    pre.set_output(transform="pandas")
    out = pre.fit_transform(X, y)
    assert list(out.columns[:2]) == ["console", "genre"]
    for column in ("console", "genre"):
        counts = X[column].value_counts()
        codes = out[column]
        assert (codes == codes.round()).all()
        assert codes.min() == 0
        # One code per frequent category plus one for the pooled rare ones.
        n_rare = int((counts < 10).sum())
        assert codes.nunique() == int((counts >= 10).sum()) + (n_rare > 0)
    query = X.head(3).assign(console="unreleased console")
    assert (pre.transform(query)["console"] == UNKNOWN_CODE).all()


def test_without_imputation_missing_values_pass_through(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = data
    pre = make_preprocessor(categorical="ordinal", impute=False)
    out = pre.set_output(transform="pandas").fit_transform(X, y)
    np.testing.assert_array_equal(out["critic_score"].isna(), X["critic_score"].isna())
    # The temporal target encoding of the earliest year has no history.
    first = (X["release_year"] == X["release_year"].min()).to_numpy()
    assert out.loc[first, "publisher_te"].isna().all()
    with pytest.raises(ValueError, match="categorical must be one of"):
        make_preprocessor(categorical="target")


def test_guard_errors() -> None:
    with pytest.raises(TypeError):
        LeakageGuard().fit(np.zeros((2, 2)))
    empty = FeatureConfig(categorical=(), high_cardinality=(), numeric=())
    with pytest.raises(ValueError, match="no columns"):
        make_preprocessor(empty)
