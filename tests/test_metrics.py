import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)

from vgsales.evaluation.metrics import (
    GREATER_IS_BETTER,
    LOG_METRICS,
    METRIC_NAMES,
    METRICS,
    ORIGINAL_METRICS,
    PRIMARY_METRIC,
    mae_log,
    regression_metrics,
    rmse,
    rmsle,
)


@pytest.fixture(scope="module")
def sales() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    y_true = rng.lognormal(mean=-2.0, sigma=1.2, size=400)
    y_pred = y_true * rng.lognormal(mean=0.0, sigma=0.5, size=400)
    return y_true, y_pred


def test_metric_groups_are_consistent() -> None:
    assert (*LOG_METRICS, *ORIGINAL_METRICS) == METRIC_NAMES
    assert set(METRICS) == set(METRIC_NAMES)
    assert PRIMARY_METRIC in LOG_METRICS
    assert set(METRIC_NAMES) >= GREATER_IS_BETTER


def test_perfect_predictions(sales: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, _ = sales
    scores = regression_metrics(y_true, y_true)
    for name in METRIC_NAMES:
        expected = 1.0 if name in GREATER_IS_BETTER else 0.0
        assert scores[name] == pytest.approx(expected)


def test_original_scale_matches_scikit_learn(
    sales: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = sales
    scores = regression_metrics(y_true, y_pred)
    assert scores["rmse"] == pytest.approx(
        math.sqrt(mean_squared_error(y_true, y_pred))
    )
    assert scores["mae"] == pytest.approx(mean_absolute_error(y_true, y_pred))
    assert scores["medae"] == pytest.approx(median_absolute_error(y_true, y_pred))
    assert scores["r2"] == pytest.approx(r2_score(y_true, y_pred))


def test_log_scale_matches_scikit_learn(
    sales: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = sales
    z_true, z_pred = np.log1p(y_true), np.log1p(y_pred)
    scores = regression_metrics(y_true, y_pred)
    assert scores["rmsle"] == pytest.approx(
        math.sqrt(mean_squared_log_error(y_true, y_pred))
    )
    assert scores["mae_log"] == pytest.approx(mean_absolute_error(z_true, z_pred))
    assert scores["medae_log"] == pytest.approx(median_absolute_error(z_true, z_pred))
    assert scores["r2_log"] == pytest.approx(r2_score(z_true, z_pred))


def test_hand_computed_values() -> None:
    y_true = np.array([0.0, 1.0, 3.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(5 / 3))
    assert rmsle(y_true, y_pred) == pytest.approx(
        math.sqrt((math.log(2) ** 2 + math.log(2) ** 2) / 3)
    )
    assert mae_log(y_true, y_pred) == pytest.approx(2 * math.log(2) / 3)


def test_log_metrics_are_scale_free_in_one_plus_sales() -> None:
    y_true = np.array([0.5, 2.0, 9.0])
    # A constant factor on 1 + y is a constant shift on the log scale.
    y_pred = 2.0 * (1.0 + y_true) - 1.0
    assert rmsle(y_true, y_pred) == pytest.approx(math.log(2))


def test_negative_predictions_are_clipped_at_zero() -> None:
    y_true = np.array([0.0, 0.2, 1.0])
    clipped = regression_metrics(y_true, np.array([-0.5, 0.2, 1.0]))
    zero = regression_metrics(y_true, np.array([0.0, 0.2, 1.0]))
    assert clipped == pytest.approx(zero)
    assert clipped["rmsle"] == 0.0


def test_individual_log_metrics_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rmsle([0.1, 0.2], [0.1, -0.2])


def test_negative_observed_sales_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        regression_metrics([-0.1, 0.2], [0.1, 0.2])


def test_constant_target_gives_nan_r2() -> None:
    scores = regression_metrics([0.3, 0.3, 0.3], [0.2, 0.3, 0.4])
    assert math.isnan(scores["r2"])
    assert math.isnan(scores["r2_log"])
    assert scores["mae"] == pytest.approx(0.2 / 3)


def test_subset_and_order_of_metrics(sales: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = sales
    scores = regression_metrics(y_true, y_pred, metrics=("mae", "rmsle"))
    assert list(scores) == ["mae", "rmsle"]


def test_accepts_pandas_series(sales: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = sales
    index = np.arange(y_true.size) + 50
    from_series = regression_metrics(
        pd.Series(y_true, index=index), pd.Series(y_pred, index=index[::-1])
    )
    assert from_series == pytest.approx(regression_metrics(y_true, y_pred))


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([0.1, 0.2], [0.1], "different lengths"),
        ([], [], "empty"),
        ([0.1, np.nan], [0.1, 0.2], "finite"),
        ([[0.1]], [[0.1]], "one-dimensional"),
    ],
)
def test_invalid_inputs(y_true: object, y_pred: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        regression_metrics(y_true, y_pred)


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown metrics"):
        regression_metrics([0.1], [0.1], metrics=("mape",))
