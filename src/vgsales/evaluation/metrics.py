r"""Regression metrics on the log and original scales.

Let :math:`y_i \ge 0` be the observed lifetime sales of row :math:`i` and
:math:`\hat y_i \ge 0` its prediction, both in millions of units, for
:math:`i = 1, \dots, n`. Write :math:`z_i = \log(1 + y_i)` and
:math:`\hat z_i = \log(1 + \hat y_i)`, and let :math:`r_i = \hat z_i - z_i`
on the log scale and :math:`e_i = \hat y_i - y_i` on the original scale.

Log scale (relative error, the primary scale for a heavy-tailed target):

- ``rmsle``: :math:`\sqrt{n^{-1} \sum_i r_i^2}`.
- ``mae_log``: :math:`n^{-1} \sum_i |r_i|`.
- ``medae_log``: :math:`\operatorname{median}_i |r_i|`.
- ``r2_log``: :math:`1 - \sum_i r_i^2 / \sum_i (z_i - \bar z)^2`.

Original scale (absolute error in millions of units, dominated by the few
best-selling titles):

- ``rmse``: :math:`\sqrt{n^{-1} \sum_i e_i^2}`.
- ``mae``: :math:`n^{-1} \sum_i |e_i|`.
- ``medae``: :math:`\operatorname{median}_i |e_i|`.
- ``r2``: :math:`1 - \sum_i e_i^2 / \sum_i (y_i - \bar y)^2`.

The coefficient of determination is undefined when the observed values are
constant; it is then reported as ``NaN`` rather than an arbitrary constant.
When :math:`\hat z` estimates :math:`E[z \mid x]`, the back-transformed
prediction :math:`\exp(\hat z) - 1` targets the geometric mean of
:math:`1 + y` minus one, which by Jensen's inequality does not exceed the
conditional mean of :math:`y`. A model fitted on the log scale therefore tends
to underpredict on the original scale. Both groups of metrics are reported so
that this trade-off is visible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

PRIMARY_METRIC: Final = "rmsle"
LOG_METRICS: Final = ("rmsle", "mae_log", "medae_log", "r2_log")
ORIGINAL_METRICS: Final = ("rmse", "mae", "medae", "r2")
METRIC_NAMES: Final = (*LOG_METRICS, *ORIGINAL_METRICS)
GREATER_IS_BETTER: Final = frozenset({"r2_log", "r2"})


def _as_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred have different lengths: {true.size} and {pred.size}"
        )
    if not (np.isfinite(true).all() and np.isfinite(pred).all()):
        raise ValueError("y_true and y_pred must be finite")
    return true, pred


def _as_sales_pair(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    true, pred = _as_pair(y_true, y_pred)
    if (true < 0).any() or (pred < 0).any():
        raise ValueError("sales values must be non-negative")
    return true, pred


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    total = float(np.sum((true - true.mean()) ** 2))
    if total == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((pred - true) ** 2)) / total


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    true, pred = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    true, pred = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def medae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Median absolute error."""
    true, pred = _as_pair(y_true, y_pred)
    return float(np.median(np.abs(pred - true)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination, ``NaN`` when ``y_true`` is constant."""
    return _r2(*_as_pair(y_true, y_pred))


def _on_log_scale(
    metric: Callable[[ArrayLike, ArrayLike], float],
) -> Callable[[ArrayLike, ArrayLike], float]:
    def wrapped(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        true, pred = _as_sales_pair(y_true, y_pred)
        return metric(np.log1p(true), np.log1p(pred))

    return wrapped


rmsle = _on_log_scale(rmse)
rmsle.__doc__ = "Root mean squared error of log1p sales."
mae_log = _on_log_scale(mae)
mae_log.__doc__ = "Mean absolute error of log1p sales."
medae_log = _on_log_scale(medae)
medae_log.__doc__ = "Median absolute error of log1p sales."
r2_log = _on_log_scale(r2)
r2_log.__doc__ = "Coefficient of determination of log1p sales."

METRICS: Final[dict[str, Callable[[ArrayLike, ArrayLike], float]]] = {
    "rmsle": rmsle,
    "mae_log": mae_log,
    "medae_log": medae_log,
    "r2_log": r2_log,
    "rmse": rmse,
    "mae": mae,
    "medae": medae,
    "r2": r2,
}


def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metrics: tuple[str, ...] = METRIC_NAMES,
) -> dict[str, float]:
    """Compute regression metrics for sales predictions.

    Predictions are clipped at zero before scoring. Sales cannot be negative,
    and a model trained on the original scale may still predict small
    negative values; the projection is applied identically to every model so
    that the log-scale metrics are always defined.

    Parameters
    ----------
    y_true : array-like of float
        Observed sales on the original scale, non-negative.
    y_pred : array-like of float
        Predicted sales on the original scale.
    metrics : tuple of str, optional
        Names from :data:`METRIC_NAMES`. Defaults to all of them.

    Returns
    -------
    dict of str to float
        Metric values in the order of ``metrics``.

    Raises
    ------
    ValueError
        If a metric name is unknown, the inputs differ in length, are empty,
        contain non-finite values, or ``y_true`` has negative values.
    """
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}; choose from {METRIC_NAMES}")
    true, pred = _as_pair(y_true, y_pred)
    if (true < 0).any():
        raise ValueError("observed sales must be non-negative")
    pred = np.clip(pred, 0.0, None)
    return {name: METRICS[name](true, pred) for name in metrics}
