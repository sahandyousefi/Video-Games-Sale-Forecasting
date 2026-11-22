import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.schema import TARGET
from vgsales.data.splits import ExpandingWindowSplit, Fold, expanding_window_folds
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.evaluation.cv import (
    AGGREGATE_ROWS,
    CVResult,
    comparison_table,
    cross_validate,
)
from vgsales.evaluation.metrics import METRIC_NAMES, regression_metrics
from vgsales.features.pipeline import make_preprocessor
from vgsales.features.prerelease import GROUP_COLUMN, build_prerelease_features
from vgsales.models.target import with_target_transform


class MaxTrainYear(RegressorMixin, BaseEstimator):
    """Predict the latest release year seen during fitting."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MaxTrainYear":
        self.max_year_ = int(X["release_year"].max())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), float(self.max_year_))


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    raw = make_synthetic(SyntheticSpec(n_titles=500), seed=8)
    scoped, _ = apply_release_scope(clean(raw)[0], load_config().data)
    scoped = scoped.set_index(scoped.index + 10_000)
    return build_prerelease_features(scoped), scoped[TARGET].to_numpy()


@pytest.fixture(scope="module")
def splitter() -> ExpandingWindowSplit:
    return ExpandingWindowSplit(n_splits=3, min_train_years=5)


@pytest.fixture(scope="module")
def ridge_result(
    data: tuple[pd.DataFrame, np.ndarray], splitter: ExpandingWindowSplit
) -> CVResult:
    X, y = data
    model = make_pipeline(make_preprocessor(scale=True), Ridge(alpha=1.0))
    return cross_validate(model, X, y, splitter)


def test_per_fold_table(ridge_result: CVResult) -> None:
    table = ridge_result.per_fold
    assert list(table["fold"]) == [0, 1, 2]
    assert {"val_start", "n_train", "n_val", "fit_seconds"} <= set(table.columns)
    assert list(table.columns[-len(METRIC_NAMES) :]) == list(METRIC_NAMES)
    assert np.isfinite(table.loc[:, list(METRIC_NAMES)].to_numpy()).all()


def test_predictions_are_out_of_fold(
    ridge_result: CVResult,
    data: tuple[pd.DataFrame, np.ndarray],
    splitter: ExpandingWindowSplit,
) -> None:
    X, y = data
    preds = ridge_result.predictions
    folds = splitter.folds(X)
    assert preds.index.is_unique
    assert len(preds) == sum(f.val_idx.size for f in folds)
    for k, fold in enumerate(folds):
        block = preds[preds["fold"] == k]
        assert block.index.equals(X.index[fold.val_idx])
        np.testing.assert_allclose(block["y_true"], y[fold.val_idx])
    # expm1 of a finite log-scale prediction is always above -1.
    assert (preds["y_pred"] > -1).all()


def test_summary_aggregates(ridge_result: CVResult) -> None:
    summary = ridge_result.summary()
    assert tuple(summary.index) == AGGREGATE_ROWS
    assert list(summary.columns) == list(METRIC_NAMES)
    scores = ridge_result.per_fold.loc[:, list(METRIC_NAMES)]
    np.testing.assert_allclose(summary.loc["mean"], scores.mean())
    np.testing.assert_allclose(summary.loc["std"], scores.std(ddof=1))
    preds = ridge_result.predictions
    pooled = regression_metrics(preds["y_true"], preds["y_pred"])
    np.testing.assert_allclose(summary.loc["pooled"], pd.Series(pooled))


def test_matches_manual_fit_on_one_fold(
    ridge_result: CVResult,
    data: tuple[pd.DataFrame, np.ndarray],
    splitter: ExpandingWindowSplit,
) -> None:
    X, y = data
    fold = splitter.folds(X)[1]
    model = make_pipeline(make_preprocessor(scale=True), Ridge(alpha=1.0))
    model.fit(X.iloc[fold.train_idx], np.log1p(y[fold.train_idx]))
    expected = np.expm1(model.predict(X.iloc[fold.val_idx]))
    got = ridge_result.predictions.query("fold == 1")["y_pred"].to_numpy()
    np.testing.assert_allclose(got, expected)


def test_estimator_never_sees_validation_years(
    data: tuple[pd.DataFrame, np.ndarray], splitter: ExpandingWindowSplit
) -> None:
    X, y = data
    result = cross_validate(MaxTrainYear(), X, y, splitter, target_transform="identity")
    preds = result.predictions
    for k, row in result.per_fold.iterrows():
        seen = preds.loc[preds["fold"] == k, "y_pred"].unique()
        assert seen.tolist() == [row["train_end"]]
        assert row["train_end"] < row["val_start"]


def test_target_transform_scales(
    data: tuple[pd.DataFrame, np.ndarray], splitter: ExpandingWindowSplit
) -> None:
    X, y = data
    fold = splitter.folds(X)[0]
    train = y[fold.train_idx]
    on_log = cross_validate(DummyRegressor(strategy="mean"), X, y, [fold])
    on_raw = cross_validate(
        DummyRegressor(strategy="mean"), X, y, [fold], target_transform="identity"
    )
    assert on_log.predictions["y_pred"].iloc[0] == pytest.approx(
        np.expm1(np.log1p(train).mean())
    )
    assert on_raw.predictions["y_pred"].iloc[0] == pytest.approx(train.mean())


def test_group_purging_is_applied(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = data
    grouped = ExpandingWindowSplit(
        n_splits=3, min_train_years=5, group_column=GROUP_COLUMN
    )
    result = cross_validate(DummyRegressor(), X, y, grouped)
    expected = grouped.folds(X)
    assert list(result.per_fold["n_purged"]) == [f.n_purged for f in expected]
    assert list(result.per_fold["n_train"]) == [f.train_idx.size for f in expected]


def test_single_fold_has_undefined_spread(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = data
    folds = expanding_window_folds(X["release_year"], n_splits=1)
    summary = cross_validate(DummyRegressor(), X, y, folds).summary()
    assert summary.loc["std"].isna().all()
    assert summary.loc["mean"].equals(summary.loc["min"])


def test_metric_subset(
    data: tuple[pd.DataFrame, np.ndarray], splitter: ExpandingWindowSplit
) -> None:
    X, y = data
    result = cross_validate(DummyRegressor(), X, y, splitter, metrics=("rmsle",))
    assert list(result.summary().columns) == ["rmsle"]
    assert "mae" not in result.per_fold.columns


def test_input_validation(
    data: tuple[pd.DataFrame, np.ndarray], splitter: ExpandingWindowSplit
) -> None:
    X, y = data
    with pytest.raises(TypeError, match="DataFrame"):
        cross_validate(DummyRegressor(), X.to_numpy(), y, splitter)
    with pytest.raises(ValueError, match="one-dimensional"):
        cross_validate(DummyRegressor(), X, y[:-1], splitter)
    with pytest.raises(ValueError, match="non-negative"):
        cross_validate(DummyRegressor(), X, -y, splitter)
    with pytest.raises(TypeError, match="sequence of Fold"):
        cross_validate(DummyRegressor(), X, y, [(np.arange(3), np.arange(3, 6))])
    folds = splitter.folds(X)
    with pytest.raises(ValueError, match="only used"):
        cross_validate(DummyRegressor(), X, y, folds, groups=X[GROUP_COLUMN])


def test_overlapping_validation_blocks_are_rejected(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = data
    fold = Fold(np.arange(10), np.arange(10, 20), (0, 0), (0, 0))
    with pytest.raises(ValueError, match="overlap"):
        cross_validate(DummyRegressor(), X, y, [fold, fold])
    outside = Fold(np.arange(10), np.array([len(X)]), (0, 0), (0, 0))
    with pytest.raises(ValueError, match="beyond"):
        cross_validate(DummyRegressor(), X, y, [outside])


def test_target_wrapper() -> None:
    model = LinearRegression()
    assert with_target_transform(model, "identity") is model
    wrapped = with_target_transform(model, "log1p")
    assert isinstance(wrapped, TransformedTargetRegressor)
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.expm1(0.1 * X.ravel())
    pred = wrapped.fit(X, y).predict(X)
    np.testing.assert_allclose(pred, y, rtol=1e-6, atol=1e-12)
    assert math.isclose(wrapped.regressor_.coef_[0], 0.1, rel_tol=1e-6)
    with pytest.raises(ValueError, match="transform"):
        with_target_transform(model, "sqrt")


def test_comparison_table_sorts_by_primary_metric(
    data: tuple[pd.DataFrame, np.ndarray], splitter: ExpandingWindowSplit
) -> None:
    X, y = data
    results = {
        "dummy": cross_validate(DummyRegressor(), X, y, splitter),
        "ridge": cross_validate(
            make_pipeline(make_preprocessor(scale=True), Ridge()), X, y, splitter
        ),
    }
    table = comparison_table(results)
    assert table.index.name == "model"
    assert set(table.index) == {"dummy", "ridge"}
    assert table["rmsle_pooled"].is_monotonic_increasing
    assert {"rmsle_mean", "rmsle_std", "r2_pooled"} <= set(table.columns)
    ridge = results["ridge"].summary()
    assert table.loc["ridge", "mae_mean"] == pytest.approx(ridge.loc["mean", "mae"])
