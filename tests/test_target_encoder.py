import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.features.encoders import TemporalTargetEncoder


@pytest.fixture
def train() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame(
        {
            "publisher": ["A", "A", "B", "A", "B", "C"],
            "release_year": [2000, 2000, 2000, 2001, 2001, 2002],
        }
    )
    y = np.array([1.0, 1.0, 4.0, 5.0, 4.0, 6.0])
    return X, y


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    raw = make_synthetic(SyntheticSpec(n_titles=500), seed=11)
    scoped, _ = apply_release_scope(clean(raw)[0], load_config().data)
    return scoped


def _encoder(smoothing: float = 1.0) -> TemporalTargetEncoder:
    return TemporalTargetEncoder(columns=["publisher"], smoothing=smoothing)


def test_hand_computed_encoding(train: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = train
    out = _encoder().fit_transform(X, y)[:, 0]
    # 2000 has no history; 2001 uses the 2000 rows (prior 2); 2002 uses both.
    expected = [np.nan, np.nan, np.nan, 4 / 3, 3.0, 3.0]
    np.testing.assert_allclose(out, expected)


def test_later_years_use_all_training_rows(
    train: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = train
    encoder = _encoder().fit(X, y)
    new = pd.DataFrame(
        {"publisher": ["A", "Z", "A"], "release_year": [2003, 2003, 1999]}
    )
    # Prior 21 / 6 = 3.5; A has sum 7 over 3 rows.
    np.testing.assert_allclose(encoder.transform(new)[:, 0], [2.625, 3.5, np.nan])


def test_zero_smoothing_gives_mean_of_earlier_rows(
    train: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = train
    out = _encoder(smoothing=0.0).fit_transform(X, y)[:, 0]
    np.testing.assert_allclose(out[3:], [1.0, 4.0, 3.0])


def test_large_smoothing_shrinks_to_prior(
    train: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = train
    out = _encoder(smoothing=1e9).fit_transform(X, y)[:, 0]
    np.testing.assert_allclose(out[3:], [2.0, 2.0, 3.0], rtol=1e-6)


def test_rows_of_the_same_year_do_not_inform_each_other(
    train: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = train
    changed = y.copy()
    changed[3:5] = [50.0, -10.0]
    base = _encoder().fit_transform(X, y)
    shifted = _encoder().fit_transform(X, changed)
    np.testing.assert_allclose(shifted[:5], base[:5])
    assert not np.isclose(shifted[5, 0], base[5, 0])


def test_future_rows_leave_past_encodings_unchanged(frame: pd.DataFrame) -> None:
    columns = ["publisher", "developer", "release_year"]
    y = np.log1p(frame["total_sales"].to_numpy())
    past = (frame["release_year"] <= 2012).to_numpy()
    encoder = TemporalTargetEncoder(smoothing=5.0)
    on_past = clone(encoder).fit(frame.loc[past, columns], y[past])
    on_all = clone(encoder).fit(frame[columns], y)
    np.testing.assert_allclose(
        on_all.transform(frame.loc[past, columns]),
        on_past.transform(frame.loc[past, columns]),
    )


def test_missing_values_form_their_own_category() -> None:
    X = pd.DataFrame(
        {"publisher": [None, np.nan, "A"], "release_year": [2000, 2000, 2000]}
    )
    encoder = _encoder(smoothing=0.0).fit(X, [2.0, 4.0, 9.0])
    new = pd.DataFrame({"publisher": [np.nan, "A"], "release_year": [2001, 2001]})
    np.testing.assert_allclose(encoder.transform(new)[:, 0], [3.0, 9.0])


def test_feature_names_and_pandas_output(
    train: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = train
    X = X.assign(developer=list("xyzxyz"))
    encoder = TemporalTargetEncoder().set_output(transform="pandas").fit(X, y)
    out = encoder.transform(X)
    assert list(out.columns) == ["publisher_te", "developer_te"]
    pd.testing.assert_index_equal(out.index, X.index)
    assert encoder.n_features_in_ == 3


def test_from_config_uses_configured_smoothing() -> None:
    config = load_config()
    encoder = TemporalTargetEncoder.from_config(
        config.features.target_encoding_smoothing
    )
    assert encoder.smoothing == config.features.target_encoding_smoothing
    assert tuple(encoder.columns) == ("publisher", "developer")


def test_errors(train: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = train
    with pytest.raises(NotFittedError):
        _encoder().transform(X)
    with pytest.raises(TypeError):
        _encoder().fit(X.to_numpy(), y)
    with pytest.raises(ValueError, match="requires y"):
        _encoder().fit(X)
    with pytest.raises(ValueError, match="shape"):
        _encoder().fit(X, y[:-1])
    with pytest.raises(ValueError, match="finite"):
        _encoder().fit(X, np.where(np.arange(6) == 2, np.nan, y))
    with pytest.raises(ValueError, match="non-negative"):
        _encoder(smoothing=-1.0).fit(X, y)
    with pytest.raises(ValueError, match="release_year"):
        _encoder().fit(X[["publisher"]], y)
    with pytest.raises(ValueError, match="missing values"):
        _encoder().fit(X.assign(release_year=[2000, None, 2001, 2001, 2002, 2002]), y)


def test_validation_rows_are_encoded_inside_cross_validation(
    frame: pd.DataFrame,
) -> None:
    X = frame[["publisher", "developer", "release_year"]]
    y = np.log1p(frame["total_sales"].to_numpy())
    cv = ExpandingWindowSplit(n_splits=3, min_train_years=5)
    model = make_pipeline(
        TemporalTargetEncoder(smoothing=10.0), SimpleImputer(), Ridge()
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    assert np.isfinite(scores).all()
    for train_idx, val_idx in cv.split(X):
        encoder = TemporalTargetEncoder().fit(X.iloc[train_idx], y[train_idx])
        assert np.isfinite(encoder.transform(X.iloc[val_idx])).all()
        first_year = X["release_year"].iloc[train_idx].min()
        is_first = (X["release_year"].iloc[train_idx] == first_year).to_numpy()
        encoded_train = encoder.transform(X.iloc[train_idx])
        assert np.isnan(encoded_train[is_first]).all()
        assert np.isfinite(encoded_train[~is_first]).all()
