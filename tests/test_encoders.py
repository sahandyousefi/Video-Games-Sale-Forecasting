import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.features.encoders import FrequencyEncoder


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "console": ["PS4", "PS4", "PS4", "NS", None],
            "genre": ["Action", "Sports", "Action", "Action", "Racing"],
        }
    )


def test_frequencies_are_training_proportions(train: pd.DataFrame) -> None:
    out = FrequencyEncoder().fit_transform(train)
    np.testing.assert_allclose(out[:, 0], [0.6, 0.6, 0.6, 0.2, 0.2])
    np.testing.assert_allclose(out[:, 1], [0.6, 0.2, 0.6, 0.6, 0.2])


def test_counts_when_not_normalised(train: pd.DataFrame) -> None:
    out = FrequencyEncoder(columns=["console"], normalize=False).fit_transform(train)
    np.testing.assert_allclose(out[:, 0], [3, 3, 3, 1, 1])


def test_unseen_categories_get_unseen_value(train: pd.DataFrame) -> None:
    encoder = FrequencyEncoder(unseen_value=-1.0).fit(train)
    new = pd.DataFrame({"console": ["PS5", "NS"], "genre": ["Puzzle", "Action"]})
    np.testing.assert_allclose(encoder.transform(new), [[-1.0, -1.0], [0.2, 0.6]])


def test_missing_values_form_their_own_category(train: pd.DataFrame) -> None:
    encoder = FrequencyEncoder(columns=["console"]).fit(train)
    out = encoder.transform(pd.DataFrame({"console": [np.nan, None]}))
    np.testing.assert_allclose(out[:, 0], [0.2, 0.2])


def test_transform_ignores_statistics_of_new_rows(train: pd.DataFrame) -> None:
    encoder = FrequencyEncoder().fit(train)
    batch = pd.DataFrame({"console": ["NS"] * 50, "genre": ["Racing"] * 50})
    np.testing.assert_allclose(encoder.transform(batch)[0], [0.2, 0.2])
    combined = encoder.transform(pd.concat([train, batch], ignore_index=True))
    np.testing.assert_allclose(combined[: len(train)], encoder.transform(train))


def test_feature_names_and_pandas_output(train: pd.DataFrame) -> None:
    encoder = FrequencyEncoder().set_output(transform="pandas").fit(train)
    out = encoder.transform(train)
    assert list(out.columns) == ["console_freq", "genre_freq"]
    pd.testing.assert_index_equal(out.index, train.index)


def test_errors(train: pd.DataFrame) -> None:
    with pytest.raises(NotFittedError):
        FrequencyEncoder().transform(train)
    with pytest.raises(TypeError):
        FrequencyEncoder().fit(train.to_numpy())
    with pytest.raises(ValueError, match="publisher"):
        FrequencyEncoder(columns=["publisher"]).fit(train)
    with pytest.raises(ValueError, match="genre"):
        FrequencyEncoder().fit(train).transform(train[["console"]])


def test_pipeline_refits_encoder_inside_each_fold() -> None:
    config = load_config()
    raw = make_synthetic(SyntheticSpec(n_titles=500), seed=3)
    frame, _ = apply_release_scope(clean(raw)[0], config.data)
    columns = ["console", "genre", "publisher"]
    X = frame[[*columns, "release_year"]]
    y = np.log1p(frame["total_sales"].to_numpy())
    cv = ExpandingWindowSplit(n_splits=3, min_train_years=5)
    model = make_pipeline(FrequencyEncoder(columns=columns), Ridge())
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    for train_idx, _ in cv.split(X):
        fitted = FrequencyEncoder(columns=columns).fit(X.iloc[train_idx])
        expected = X.iloc[train_idx]["console"].value_counts(normalize=True)
        pd.testing.assert_series_equal(
            fitted.frequencies_["console"], expected, check_names=False
        )
