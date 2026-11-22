import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.schema import TARGET
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.evaluation.cv import cross_validate
from vgsales.features.prerelease import build_prerelease_features
from vgsales.models.baselines import (
    BASELINE_NAMES,
    GroupMedianRegressor,
    make_baseline,
)


@pytest.fixture
def cells() -> tuple[pd.DataFrame, np.ndarray]:
    genre = ["A"] * 6 + ["A"] * 2 + ["B"] * 5
    console = ["X"] * 6 + ["Y"] * 2 + ["X", "Y", "Y", "Z", "Z"]
    y = np.array([1, 2, 3, 4, 5, 6, 10, 20, 7, 8, 9, 30, 40], dtype=float)
    return pd.DataFrame({"genre": genre, "console": console}), y


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    raw = make_synthetic(SyntheticSpec(n_titles=800), seed=5)
    scoped, _ = apply_release_scope(clean(raw)[0], load_config().data)
    return build_prerelease_features(scoped), scoped[TARGET].to_numpy()


def test_group_median_backs_off_to_coarser_cells(
    cells: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = cells
    model = GroupMedianRegressor(min_count=5).fit(X, y)
    query = pd.DataFrame(
        {
            "genre": ["A", "A", "B", "C", None],
            "console": ["X", "Y", "X", "X", "X"],
        }
    )
    pred = model.predict(query)
    # A/X has six rows; A/Y has two and backs off to genre A; B has five rows
    # in total but no cell of five; C is unseen.
    assert pred[0] == pytest.approx(3.5)
    assert pred[1] == pytest.approx(np.median(y[:8]))
    assert pred[2] == pytest.approx(np.median(y[8:]))
    assert pred[3] == pytest.approx(np.median(y))
    assert pred[4] == pytest.approx(np.median(y))
    np.testing.assert_array_equal(model.backoff_level(query), [2, 1, 1, 0, 0])


def test_group_median_with_min_count_one_uses_every_cell(
    cells: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = cells
    model = GroupMedianRegressor(min_count=1).fit(X, y)
    np.testing.assert_array_equal(model.backoff_level(X), np.full(len(X), 2))
    assert model.predict(X.iloc[[6]])[0] == pytest.approx(15.0)


def test_missing_categories_form_their_own_cell() -> None:
    X = pd.DataFrame({"genre": [None, None, "A"], "console": ["X", "X", "X"]})
    model = GroupMedianRegressor(min_count=2).fit(X, np.array([1.0, 3.0, 9.0]))
    assert model.predict(X.iloc[[0]])[0] == pytest.approx(2.0)


def test_group_median_is_clonable_and_validates(
    cells: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = cells
    model = clone(GroupMedianRegressor(columns=("genre",), min_count=3))
    assert model.get_params() == {"columns": ("genre",), "min_count": 3}
    with pytest.raises(TypeError, match="DataFrame"):
        model.fit(X.to_numpy(), y)
    with pytest.raises(ValueError, match="grouping columns"):
        model.fit(X.drop(columns="genre"), y)
    with pytest.raises(ValueError, match="min_count"):
        GroupMedianRegressor(min_count=0).fit(X, y)
    with pytest.raises(ValueError, match="at least one"):
        GroupMedianRegressor(columns=()).fit(X, y)


def test_empty_prediction_frame(cells: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = cells
    model = GroupMedianRegressor().fit(X, y)
    assert model.predict(X.iloc[:0]).shape == (0,)
    assert model.backoff_level(X.iloc[:0]).shape == (0,)


def test_global_median_predicts_training_median(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = data
    model = make_baseline("global_median").fit(X, np.log1p(y))
    pred = model.predict(X.iloc[:5])
    np.testing.assert_allclose(pred, np.median(np.log1p(y)))


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baselines_reject_leakage_columns(
    name: str, data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    X, y = data
    with pytest.raises(ValueError, match="leakage"):
        make_baseline(name).fit(X.assign(na_sales=0.0), y)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baselines_cross_validate(
    name: str, data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    X, y = data
    cv = ExpandingWindowSplit(n_splits=3, min_train_years=5)
    model = make_baseline(name, load_config().features)
    result = cross_validate(model, X, y, cv)
    assert np.isfinite(result.predictions["y_pred"]).all()
    assert np.isfinite(result.summary().loc["mean", "rmsle"])


def test_informed_baselines_beat_the_global_median_on_synthetic_data(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = data
    cv = ExpandingWindowSplit(n_splits=3, min_train_years=5)
    pooled = {
        name: cross_validate(make_baseline(name), X, y, cv)
        .summary()
        .loc["pooled", "rmsle"]
        for name in BASELINE_NAMES
    }
    assert pooled["ridge"] < pooled["global_median"]
    assert pooled["group_median"] <= pooled["global_median"] * 1.05


def test_unknown_baseline() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        make_baseline("mean")
