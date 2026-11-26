import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from vgsales.config import load_config
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.evaluation.cv import cross_validate
from vgsales.evaluation.intervals import pinball_loss
from vgsales.models.boosting import LightGBMRegressor
from vgsales.models.quantile import (
    LightGBMQuantileRegressor,
    QuantileIntervalRegressor,
    interval_levels,
    predict_sales_interval,
    rearrange_quantiles,
)
from vgsales.models.registry import is_available
from vgsales.models.target import with_target_transform

FAST = {"n_estimators": 200, "early_stopping_rounds": 20}

needs_lightgbm = pytest.mark.skipif(
    not is_available("lightgbm"), reason="lightgbm is not installed"
)


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    config = load_config()
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=900), seed=6), config
    )
    X, y = dataset.development()
    X_test, y_test = dataset.holdout()
    return X, np.log1p(y), X_test, np.log1p(y_test)


def test_interval_levels() -> None:
    assert interval_levels(0.1) == pytest.approx((0.05, 0.5, 0.95))
    assert interval_levels(0.5) == pytest.approx((0.25, 0.5, 0.75))
    for alpha in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="alpha"):
            interval_levels(alpha)


def test_rearrangement_sorts_each_row() -> None:
    crossed = np.array([[1.0, 0.5, 2.0], [0.0, 1.0, 0.5], [0.1, 0.2, 0.3]])
    expected = np.array([[0.5, 1.0, 2.0], [0.0, 0.5, 1.0], [0.1, 0.2, 0.3]])
    assert np.array_equal(rearrange_quantiles(crossed), expected)
    assert crossed[0, 0] == 1.0
    with pytest.raises(ValueError, match="two-dimensional"):
        rearrange_quantiles([1.0, 2.0])


@needs_lightgbm
def test_quantile_model_uses_the_pinball_loss(data) -> None:
    X, z, X_test, z_test = data
    upper = LightGBMQuantileRegressor(quantile=0.9, **FAST).fit(X, z)
    params = upper.booster_.get_params()
    assert params["objective"] == "quantile"
    assert params["metric"] == "quantile"
    assert params["alpha"] == pytest.approx(0.9)
    assert upper.early_stopping_years_ is not None
    assert np.mean(z <= upper.predict(X)) > 0.75

    lower = LightGBMQuantileRegressor(quantile=0.1, **FAST).fit(X, z)
    assert np.mean(z >= lower.predict(X)) > 0.75
    mean = LightGBMRegressor(**FAST).fit(X, z)
    assert mean.booster_.get_params()["objective"] == "regression"
    assert pinball_loss(z_test, upper.predict(X_test), 0.9) < pinball_loss(
        z_test, lower.predict(X_test), 0.9
    )


@needs_lightgbm
def test_quantile_level_is_validated(data) -> None:
    X, z, _, _ = data
    with pytest.raises(ValueError, match="quantile"):
        LightGBMQuantileRegressor(quantile=1.5, **FAST).fit(X, z)
    with pytest.raises(ValueError, match="must not set"):
        LightGBMQuantileRegressor(params={"alpha": 0.3}, **FAST).fit(X, z)


@needs_lightgbm
def test_interval_regressor_fits_one_model_per_level(data) -> None:
    X, z, X_test, _ = data
    model = QuantileIntervalRegressor(alpha=0.2, random_state=3, **FAST).fit(X, z)
    assert model.levels_ == pytest.approx((0.1, 0.5, 0.9))
    assert [m.quantile for m in model.models_] == pytest.approx(list(model.levels_))
    assert len(model.n_estimators_) == 3
    assert all(m.random_state == 3 for m in model.models_)

    quantiles = model.predict_quantiles(X_test)
    assert quantiles.shape == (len(X_test), 3)
    assert (np.diff(quantiles, axis=1) >= 0).all()
    assert np.array_equal(model.predict(X_test), quantiles[:, 1])
    lower, upper = model.predict_interval(X_test)
    assert np.array_equal(lower, quantiles[:, 0])
    assert np.array_equal(upper, quantiles[:, 2])

    params = clone(model).get_params()
    assert params["alpha"] == 0.2
    assert not hasattr(clone(model), "models_")


@needs_lightgbm
def test_sales_intervals_are_back_transformed(data) -> None:
    X, z, X_test, _ = data
    model = with_target_transform(QuantileIntervalRegressor(**FAST), "log1p")
    model.fit(X, np.expm1(z))
    lower, upper = predict_sales_interval(model, X_test)
    z_lower, z_upper = model.regressor_.predict_interval(X_test)
    assert np.allclose(lower, np.expm1(z_lower))
    assert np.allclose(upper, np.expm1(z_upper))
    point = model.predict(X_test)
    assert (lower <= point).all()
    assert (point <= upper).all()

    bare = QuantileIntervalRegressor(**FAST).fit(X, z)
    assert np.allclose(predict_sales_interval(bare, X_test)[0], z_lower)
    with pytest.raises(TypeError, match="does not predict intervals"):
        predict_sales_interval(LightGBMRegressor(**FAST).fit(X, z), X_test)


@needs_lightgbm
def test_median_is_cross_validated_on_the_shared_folds(data) -> None:
    X, z, _, _ = data
    splitter = ExpandingWindowSplit(n_splits=3, min_train_years=5)
    result = cross_validate(QuantileIntervalRegressor(**FAST), X, np.expm1(z), splitter)
    assert len(result.per_fold) == 3
    assert np.isfinite(result.per_fold["rmsle"]).all()


def test_frame_input_is_required() -> None:
    with pytest.raises(ValueError, match="release_year"):
        QuantileIntervalRegressor().fit(pd.DataFrame({"a": [1.0]}), [0.1])
