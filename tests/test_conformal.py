import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression

from vgsales.config import config_from_dict, load_config
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.evaluation.interval_cv import cross_validate_intervals
from vgsales.features.prerelease import GROUP_COLUMN
from vgsales.models.baselines import make_baseline
from vgsales.models.conformal import (
    ConformalQuantileRegressor,
    SplitConformalRegressor,
    calibration_split,
    conformal_quantile,
    min_calibration_size,
)
from vgsales.models.intervals import (
    INTERVAL_METHODS,
    interval_dependency,
    make_interval_model,
)
from vgsales.models.quantile import QuantileIntervalRegressor, predict_sales_interval
from vgsales.models.registry import is_available
from vgsales.models.target import with_target_transform


class OnX(RegressorMixin, BaseEstimator):
    """Linear regression on the column ``x`` that records its training years."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OnX":
        self.model_ = LinearRegression().fit(X[["x"]], y)
        self.years_ = (int(X["release_year"].min()), int(X["release_year"].max()))
        self.n_rows_ = len(X)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[["x"]])


class Band(OnX):
    """Regression on ``x`` with a band of fixed half-width around it."""

    def __init__(self, alpha: float = 0.1, half_width: float = 0.1) -> None:
        self.alpha = alpha
        self.half_width = half_width

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        center = self.predict(X)
        return center - self.half_width, center + self.half_width


def make_frame(rng: np.random.Generator, n: int, years: tuple[int, int]) -> tuple:
    x = rng.normal(size=n)
    frame = pd.DataFrame(
        {"x": x, "release_year": rng.integers(years[0], years[1] + 1, size=n)}
    )
    return frame, 1.0 + 0.5 * x + rng.standard_t(df=4, size=n) * 0.3


def test_conformal_quantile_rank() -> None:
    scores = np.arange(19, 0, -1, dtype=float)
    assert conformal_quantile(scores, 0.1) == 18.0
    assert conformal_quantile(scores, 0.5) == 10.0
    assert conformal_quantile(np.arange(1.0, 10.0), 0.1) == 9.0
    assert min_calibration_size(0.1) == 9
    assert min_calibration_size(0.05) == 19
    with pytest.raises(ValueError, match="at least 9"):
        conformal_quantile(np.arange(1.0, 9.0), 0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        conformal_quantile(scores, 0.0)
    with pytest.raises(ValueError, match="finite"):
        conformal_quantile([1.0, np.nan], 0.5)


def test_marginal_coverage_on_exchangeable_data() -> None:
    rng = np.random.default_rng(0)
    covered, bounds = [], []
    for _ in range(200):
        X, z = make_frame(rng, 120, (2000, 2005))
        model = SplitConformalRegressor(OnX(), alpha=0.2, calibration_years=2).fit(X, z)
        X_new, z_new = make_frame(rng, 200, (2006, 2006))
        lower, upper = model.predict_interval(X_new)
        covered.append(np.mean((z_new >= lower) & (z_new <= upper)))
        bounds.append(1 - 0.2 + 1 / (model.n_calibration_ + 1))
    # The mean over 200 replications has a standard error near 0.004.
    assert 0.8 - 0.015 <= np.mean(covered) <= np.mean(bounds) + 0.015


def test_point_model_never_sees_the_calibration_years() -> None:
    rng = np.random.default_rng(1)
    X, z = make_frame(rng, 300, (2000, 2009))
    model = SplitConformalRegressor(OnX(), alpha=0.1, calibration_years=3).fit(X, z)
    assert model.calibration_years_ == (2007, 2009)
    assert model.estimator_.years_[1] == 2006
    assert model.n_calibration_ == int((X["release_year"] >= 2007).sum())
    assert model.n_train_ + model.n_calibration_ == len(X)
    assert model.scores_.shape == (model.n_calibration_,)
    assert model.quantile_ == conformal_quantile(model.scores_, 0.1)

    lower, upper = model.predict_interval(X)
    center = model.predict(X)
    assert np.allclose(center - lower, model.quantile_)
    assert np.allclose(upper - center, model.quantile_)


def test_titles_with_calibration_rows_are_purged() -> None:
    rng = np.random.default_rng(2)
    X, z = make_frame(rng, 200, (2000, 2009))
    X[GROUP_COLUMN] = [f"t{i}" for i in range(len(X))]
    late = X.index[X["release_year"] == 2009][0]
    early = X.index[X["release_year"] == 2001][0]
    X.loc[early, GROUP_COLUMN] = X.loc[late, GROUP_COLUMN]
    train_idx, cal_idx = calibration_split(X, 2)
    assert early not in X.index[train_idx]
    assert early not in X.index[cal_idx]
    assert late in X.index[cal_idx]
    model = SplitConformalRegressor(OnX(), calibration_years=2).fit(X, z)
    assert model.estimator_.n_rows_ == len(X) - model.n_calibration_ - 1


def test_invalid_settings() -> None:
    rng = np.random.default_rng(3)
    X, z = make_frame(rng, 100, (2000, 2002))
    with pytest.raises(ValueError, match="point estimator"):
        SplitConformalRegressor().fit(X, z)
    with pytest.raises(ValueError, match="calibration set"):
        SplitConformalRegressor(OnX(), calibration_years=3).fit(X, z)
    with pytest.raises(ValueError, match=">= 1"):
        SplitConformalRegressor(OnX(), calibration_years=0).fit(X, z)
    with pytest.raises(TypeError, match="integer"):
        SplitConformalRegressor(OnX(), calibration_years=1.5).fit(X, z)
    with pytest.raises(ValueError, match="between 0 and 1"):
        SplitConformalRegressor(OnX(), alpha=1.5).fit(X, z)
    with pytest.raises(ValueError, match="release_year"):
        SplitConformalRegressor(OnX()).fit(X.drop(columns="release_year"), z)
    small, z_small = make_frame(rng, 40, (2000, 2010))
    with pytest.raises(ValueError, match="too few"):
        SplitConformalRegressor(OnX(), alpha=0.01, calibration_years=1).fit(
            small, z_small
        )
    params = clone(SplitConformalRegressor(OnX(), alpha=0.3)).get_params()
    assert params["alpha"] == 0.3
    assert params["calibration_years"] == 2


def test_cqr_scores_and_adjusted_bounds() -> None:
    rng = np.random.default_rng(5)
    X, z = make_frame(rng, 300, (2000, 2009))
    model = ConformalQuantileRegressor(Band(half_width=0.1), alpha=0.2).fit(X, z)
    assert model.estimator_.alpha == 0.2
    assert model.estimator.alpha == 0.1
    cal = (X["release_year"] >= 2008).to_numpy()
    center = model.estimator_.predict(X[cal])
    expected = np.maximum(center - 0.1 - z[cal], z[cal] - center - 0.1)
    assert np.allclose(model.scores_, expected)
    assert model.quantile_ == conformal_quantile(expected, 0.2)
    assert model.quantile_ > 0

    lower, upper = model.predict_interval(X)
    point = model.predict(X)
    assert np.allclose(lower, point - 0.1 - model.quantile_)
    assert np.allclose(upper, point + 0.1 + model.quantile_)


def test_cqr_shrinks_wide_intervals_without_emptying_them() -> None:
    rng = np.random.default_rng(6)
    X, z = make_frame(rng, 300, (2000, 2009))
    model = ConformalQuantileRegressor(Band(half_width=5.0), alpha=0.2).fit(X, z)
    assert model.quantile_ < 0
    lower, upper = model.predict_interval(X)
    point = model.predict(X)
    assert np.allclose(upper - lower, 2 * (5.0 + model.quantile_))
    assert (lower <= point).all()
    assert (point <= upper).all()

    # A test row whose band is narrower than 2|Q| collapses to its point forecast.
    model.estimator_.half_width = 0.5
    assert 0.5 + model.quantile_ < 0
    lower, upper = model.predict_interval(X)
    assert np.allclose(lower, point)
    assert np.allclose(upper, point)


def test_cqr_restores_coverage_of_a_narrow_band() -> None:
    rng = np.random.default_rng(7)
    raw, calibrated = [], []
    for _ in range(100):
        X, z = make_frame(rng, 150, (2000, 2005))
        model = ConformalQuantileRegressor(Band(), alpha=0.2).fit(X, z)
        X_new, z_new = make_frame(rng, 200, (2006, 2006))
        lower, upper = model.estimator_.predict_interval(X_new)
        raw.append(np.mean((z_new >= lower) & (z_new <= upper)))
        lower, upper = model.predict_interval(X_new)
        calibrated.append(np.mean((z_new >= lower) & (z_new <= upper)))
    assert np.mean(raw) < 0.4
    assert 0.78 <= np.mean(calibrated) <= 0.86


def test_cqr_needs_an_interval_model() -> None:
    rng = np.random.default_rng(8)
    X, z = make_frame(rng, 100, (2000, 2005))
    with pytest.raises(TypeError, match="does not predict intervals"):
        ConformalQuantileRegressor(OnX()).fit(X, z)
    with pytest.raises(ValueError, match="point estimator"):
        ConformalQuantileRegressor().fit(X, z)


@pytest.fixture(scope="module")
def dataset():
    return prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=600), seed=4), load_config()
    )


def test_sales_intervals_are_multiplicative_bands(dataset) -> None:
    X, y = dataset.development()
    X_test, _ = dataset.holdout()
    point = make_baseline("ridge", load_config().features, alpha=1.0)
    model = with_target_transform(SplitConformalRegressor(point), "log1p").fit(X, y)
    lower, upper = predict_sales_interval(model, X_test)
    y_hat = model.predict(X_test)
    q = model.regressor_.quantile_
    assert np.allclose(lower, (1 + y_hat) * np.exp(-q) - 1)
    assert np.allclose(upper, (1 + y_hat) * np.exp(q) - 1)


def test_conformal_intervals_on_the_folds(dataset) -> None:
    X, y = dataset.development()
    config = config_from_dict({"intervals": {"point_model": "ridge"}})
    model = make_interval_model("conformal", config)
    splitter = ExpandingWindowSplit(n_splits=3, min_train_years=6)
    result = cross_validate_intervals(model, X, y, splitter, alpha=0.1)
    assert len(result.per_fold) == 3
    widths = result.per_fold["width_log"]
    assert (widths > 0).all()
    assert 0.0 < result.summary().loc["pooled", "coverage"] <= 1.0


def test_interval_methods_registry() -> None:
    config = load_config()
    assert INTERVAL_METHODS == ("quantile", "conformal", "cqr")
    assert set(config.intervals.methods) <= set(INTERVAL_METHODS)
    ridge = config_from_dict({"intervals": {"point_model": "ridge"}})
    assert interval_dependency("conformal", ridge) is None
    model = make_interval_model("conformal", ridge)
    assert isinstance(model, SplitConformalRegressor)
    assert model.alpha == ridge.intervals.alpha
    assert model.calibration_years == 2
    with pytest.raises(ValueError, match="unknown interval method"):
        make_interval_model("bootstrap", config)
    unknown = config_from_dict({"intervals": {"point_model": "forest"}})
    with pytest.raises(ValueError, match="unknown model"):
        make_interval_model("conformal", unknown)
    if is_available("lightgbm"):
        quantile = make_interval_model("quantile", config)
        assert isinstance(quantile, QuantileIntervalRegressor)
        assert quantile.random_state == config.seed
        cqr = make_interval_model("cqr", config)
        assert isinstance(cqr, ConformalQuantileRegressor)
        assert isinstance(cqr.estimator, QuantileIntervalRegressor)
        assert cqr.estimator.get_params() == quantile.get_params()
        assert cqr.calibration_years == config.intervals.calibration_years
    else:
        for method in ("quantile", "cqr"):
            with pytest.raises(ImportError, match="lightgbm"):
                make_interval_model(method, config)


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm is not installed")
def test_cqr_with_lightgbm_quantiles_on_the_folds(dataset) -> None:
    X, y = dataset.development()
    fast = {"n_estimators": 150, "early_stopping_rounds": 20, "validation_years": 2}
    config = config_from_dict({"intervals": {"quantile_params": fast}})
    model = make_interval_model("cqr", config)
    splitter = ExpandingWindowSplit(n_splits=2, min_train_years=8)
    result = cross_validate_intervals(model, X, y, splitter, alpha=0.1)
    frame = result.predictions
    assert (frame["lower"] <= frame["y_pred"] + 1e-9).all()
    assert (frame["y_pred"] <= frame["upper"] + 1e-9).all()
    assert 0.0 < result.summary().loc["pooled", "coverage"] <= 1.0
