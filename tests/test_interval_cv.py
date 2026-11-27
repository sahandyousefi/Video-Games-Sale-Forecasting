import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

from vgsales.config import load_config
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.evaluation.cv import FOLD_KEYS
from vgsales.evaluation.interval_cv import (
    POINT_METRICS,
    PREDICTION_COLUMNS,
    IntervalCVResult,
    compare_intervals,
    cross_validate_intervals,
    interval_comparison_table,
)
from vgsales.evaluation.intervals import INTERVAL_METRICS, interval_metrics
from vgsales.models.quantile import QuantileIntervalRegressor, make_quantile_interval
from vgsales.models.registry import is_available


class MedianBand(RegressorMixin, BaseEstimator):
    """Training median with a band of fixed half-width on the fitted scale."""

    def __init__(self, alpha: float = 0.1, half_width: float = 0.5) -> None:
        self.alpha = alpha
        self.half_width = half_width

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MedianBand":
        self.center_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.center_)

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        center = self.predict(X)
        return center - self.half_width, center + self.half_width


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=600), seed=11), load_config()
    )
    return dataset.development()


@pytest.fixture(scope="module")
def splitter() -> ExpandingWindowSplit:
    return ExpandingWindowSplit(n_splits=3, min_train_years=5)


@pytest.fixture(scope="module")
def result(data, splitter) -> IntervalCVResult:
    X, y = data
    return cross_validate_intervals(MedianBand(), X, y, splitter, alpha=0.1)


def test_per_fold_table(result: IntervalCVResult) -> None:
    assert len(result.per_fold) == 3
    assert result.metrics == (*INTERVAL_METRICS, *POINT_METRICS)
    assert result.point_metrics == POINT_METRICS
    for column in (*FOLD_KEYS, "fit_seconds", *result.metrics):
        assert column in result.per_fold.columns
    coverage = result.per_fold["coverage"]
    assert ((coverage >= 0) & (coverage <= 1)).all()
    total = result.per_fold[["coverage", "below", "above"]].sum(axis=1)
    assert np.allclose(total, 1.0)


def test_intervals_are_out_of_fold_and_back_transformed(
    data, splitter, result: IntervalCVResult
) -> None:
    X, y = data
    frame = result.predictions
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert frame.index.is_unique
    assert (frame["lower"] <= frame["y_pred"]).all()
    assert (frame["y_pred"] <= frame["upper"]).all()

    fold = splitter.folds(X)[0]
    center = np.median(np.log1p(y[fold.train_idx]))
    first = frame[frame["fold"] == 0]
    assert list(first.index) == list(X.index[fold.val_idx])
    assert np.allclose(first["lower"], np.expm1(center - 0.5))
    assert np.allclose(first["upper"], np.expm1(center + 0.5))
    assert np.allclose(first["y_true"], y[fold.val_idx])
    expected = interval_metrics(y[fold.val_idx], first["lower"], first["upper"], 0.1)
    for name, value in expected.items():
        assert result.per_fold.loc[0, name] == pytest.approx(value)


def test_summary_pools_the_out_of_fold_intervals(result: IntervalCVResult) -> None:
    summary = result.summary()
    assert list(summary.index) == ["mean", "std", "min", "max", "pooled"]
    assert list(summary.columns) == list(result.metrics)
    frame = result.predictions
    pooled = interval_metrics(frame["y_true"], frame["lower"], frame["upper"], 0.1)
    for name, value in pooled.items():
        assert summary.loc["pooled", name] == pytest.approx(value)
    # Lower bounds below zero sales are clipped, which can only narrow the band.
    assert 0.0 < summary.loc["max", "width_log"] <= 1.0 + 1e-12
    assert summary.loc["mean", "coverage"] == pytest.approx(
        result.per_fold["coverage"].mean()
    )


def test_interval_model_is_checked(data, splitter) -> None:
    X, y = data
    with pytest.raises(TypeError, match="does not predict intervals"):
        cross_validate_intervals(Ridge(), X, y, splitter, alpha=0.1)
    with pytest.raises(ValueError, match=r"alpha=0\.2"):
        cross_validate_intervals(MedianBand(alpha=0.2), X, y, splitter, alpha=0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        cross_validate_intervals(MedianBand(), X, y, splitter, alpha=1.0)
    with pytest.raises(TypeError, match="DataFrame"):
        cross_validate_intervals(MedianBand(), X.to_numpy(), y, splitter, alpha=0.1)


def test_methods_are_compared_on_the_same_folds(data, splitter) -> None:
    X, y = data
    models = {"narrow": MedianBand(half_width=0.1), "wide": MedianBand(half_width=3.0)}
    results = compare_intervals(models, X, y, splitter, alpha=0.1)
    assert list(results) == ["narrow", "wide"]
    narrow, wide = results["narrow"], results["wide"]
    assert narrow.predictions.index.equals(wide.predictions.index)
    assert (wide.per_fold["coverage"] >= narrow.per_fold["coverage"]).all()

    table = interval_comparison_table(results)
    assert table.index.name == "method"
    scores = table["interval_score_log_pooled"]
    assert scores.is_monotonic_increasing
    for metric in ("coverage", "width_log", "rmsle"):
        for stat in ("mean", "std", "pooled"):
            assert f"{metric}_{stat}" in table.columns

    mismatched = {
        "a": narrow,
        "b": cross_validate_intervals(MedianBand(alpha=0.2), X, y, splitter, alpha=0.2),
    }
    with pytest.raises(ValueError, match="different levels"):
        interval_comparison_table(mismatched)
    with pytest.raises(ValueError, match="must not be empty"):
        compare_intervals({}, X, y, splitter, alpha=0.1)


def test_builder_follows_the_interval_configuration() -> None:
    config = load_config()
    model = make_quantile_interval(
        config.intervals.alpha,
        config.features,
        random_state=config.seed,
        **config.intervals.quantile_params,
    )
    assert isinstance(model, QuantileIntervalRegressor)
    assert model.alpha == pytest.approx(0.1)
    assert model.n_estimators == 2000
    assert model.validation_years == 2
    assert model.random_state == config.seed
    assert model.params == {
        "num_leaves": 31,
        "min_child_samples": 20,
        "min_data_per_group": 50,
    }


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm is not installed")
def test_quantile_intervals_on_the_folds(data, splitter) -> None:
    X, y = data
    model = make_quantile_interval(0.2, n_estimators=150, early_stopping_rounds=20)
    result = cross_validate_intervals(model, X, y, splitter, alpha=0.2)
    frame = result.predictions
    assert len(frame) == result.per_fold["n_val"].sum()
    assert (frame["lower"] <= frame["upper"]).all()
    assert (frame["lower"] <= frame["y_pred"] + 1e-12).all()
    assert 0.0 < result.summary().loc["pooled", "coverage"] <= 1.0
