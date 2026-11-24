from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vgsales.config import FeatureConfig, load_config
from vgsales.data.cleaning import clean
from vgsales.data.schema import TARGET
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.features.prerelease import build_prerelease_features
from vgsales.features.tabular import (
    MISSING_CATEGORY,
    OOV_INDEX,
    TabularArrays,
    TabularEncoder,
)


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    raw = make_synthetic(SyntheticSpec(n_titles=800), seed=5)
    scoped, _ = apply_release_scope(clean(raw)[0], load_config().data)
    return build_prerelease_features(scoped), np.log1p(scoped[TARGET].to_numpy())


def test_output_shapes_and_dtypes(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, z = data
    encoder = TabularEncoder()
    arrays = encoder.fit_transform(X, z)
    config = FeatureConfig()
    assert encoder.embedding_columns_ == (
        *config.categorical,
        *config.high_cardinality,
    )
    assert arrays.categorical.shape == (len(X), 4)
    assert arrays.categorical.dtype == np.int64
    assert arrays.numeric.dtype == np.float32
    assert arrays.numeric.shape == (len(X), encoder.n_numeric_)
    assert np.isfinite(arrays.numeric).all()
    names = list(encoder.numeric_features_)
    assert names[:4] == [
        "publisher_freq",
        "developer_freq",
        "publisher_te",
        "developer_te",
    ]
    assert names[4:] == list(config.numeric)
    for j, size in enumerate(encoder.cardinalities_):
        assert arrays.categorical[:, j].min() >= 0
        assert arrays.categorical[:, j].max() < size


def test_rare_and_unseen_categories_share_index_zero(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    encoder = TabularEncoder(FeatureConfig(min_category_count=5)).fit(X, z)
    j = encoder.embedding_columns_.index("developer")
    counts = X["developer"].value_counts()
    rare = counts.index[counts < 5]
    assert len(rare) > 0
    codes = encoder.transform(X).categorical[:, j]
    is_rare = X["developer"].isin(rare).to_numpy()
    assert (codes[is_rare] == OOV_INDEX).all()
    assert (codes[~is_rare & X["developer"].notna().to_numpy()] > 0).all()
    assert encoder.cardinalities_[j] == (counts >= 5).sum() + 1

    unseen = X.head(3).assign(developer="Studio never seen")
    assert (encoder.transform(unseen).categorical[:, j] == OOV_INDEX).all()


def test_missing_values_form_their_own_category(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    X = X.copy()
    X.loc[X.index[:10], "publisher"] = np.nan
    encoder = TabularEncoder(FeatureConfig(min_category_count=5)).fit(X, z)
    j = encoder.embedding_columns_.index("publisher")
    assert MISSING_CATEGORY in encoder.vocabularies_[j]
    codes = encoder.transform(X.head(10)).categorical[:, j]
    assert (codes == encoder.vocabularies_[j][MISSING_CATEGORY]).all()


def test_numeric_inputs_are_standardized_on_the_training_rows(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    numeric = TabularEncoder().fit_transform(X, z).numeric.astype(np.float64)
    spread = numeric.std(axis=0)
    varying = spread > 0
    np.testing.assert_allclose(numeric.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(spread[varying], 1.0, atol=1e-4)


def test_encoding_is_row_wise(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, z = data
    encoder = TabularEncoder().fit(X, z)
    full = encoder.transform(X)
    rows = np.arange(0, len(X), 7)
    part = encoder.transform(X.iloc[rows])
    np.testing.assert_array_equal(part.categorical, full.categorical[rows])
    np.testing.assert_array_equal(part.numeric, full.numeric[rows])


def test_target_is_required_only_for_target_encodings(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, _ = data
    with pytest.raises(ValueError, match="y is required"):
        TabularEncoder().fit(X)
    encoder = TabularEncoder(high_cardinality_encodings=False).fit(X)
    assert list(encoder.numeric_features_) == list(FeatureConfig().numeric)


def test_numeric_block_may_be_empty(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, _ = data
    config = replace(FeatureConfig(), numeric=())
    arrays = TabularEncoder(config, high_cardinality_encodings=False).fit_transform(X)
    assert arrays.numeric.shape == (len(X), 0)
    assert arrays.categorical.shape == (len(X), 4)


def test_leakage_columns_are_rejected(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, z = data
    with pytest.raises(ValueError, match="na_sales"):
        TabularEncoder().fit(X.assign(na_sales=0.1), z)
    encoder = TabularEncoder().fit(X, z)
    with pytest.raises(ValueError, match="total_sales"):
        encoder.transform(X.assign(total_sales=1.0))


def test_arrays_subset_and_validation() -> None:
    arrays = TabularArrays(
        np.arange(6, dtype=np.int64).reshape(3, 2), np.zeros((3, 1), np.float32)
    )
    assert len(arrays) == 3
    part = arrays.subset([2, 0])
    np.testing.assert_array_equal(part.categorical, [[4, 5], [0, 1]])
    with pytest.raises(ValueError, match="rows"):
        TabularArrays(np.zeros((3, 2), np.int64), np.zeros((2, 1), np.float32))
