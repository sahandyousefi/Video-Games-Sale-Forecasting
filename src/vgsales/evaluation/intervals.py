r"""Scores of quantile forecasts and central prediction intervals.

Let :math:`y_i \ge 0` be the observed lifetime sales of row :math:`i`, in
millions of units, for :math:`i = 1, \dots, n`.

For a quantile level :math:`\tau \in (0, 1)` and a predicted quantile
:math:`q_i`, the pinball loss (Koenker and Bassett, 1978) is

.. math::

    \rho_\tau(y_i - q_i), \qquad \rho_\tau(u) = u \, (\tau - \mathbf{1}\{u < 0\}),

whose expectation is minimized by the :math:`\tau`-quantile of the
conditional distribution of :math:`y_i`. ``pinball_loss`` returns the mean
over rows.

A central :math:`(1 - \alpha)` prediction interval :math:`[l_i, u_i]` is
formed by the predicted :math:`\alpha / 2` and :math:`1 - \alpha / 2`
quantiles. It is scored by:

- ``coverage``: :math:`n^{-1} \sum_i \mathbf{1}\{l_i \le y_i \le u_i\}`, to
  be compared with the nominal level :math:`1 - \alpha`. The shares of
  observations ``below`` and ``above`` the interval are reported as well,
  since a symmetric and a one-sided miscoverage give the same coverage.
- ``width``: the mean width :math:`n^{-1} \sum_i (u_i - l_i)`.
- ``interval_score`` (Gneiting and Raftery, 2007): the mean of

  .. math::

      S_\alpha(l, u; y) = (u - l) + \frac{2}{\alpha} (l - y) \mathbf{1}\{y < l\}
          + \frac{2}{\alpha} (y - u) \mathbf{1}\{y > u\},

  a proper scoring rule that rewards narrow intervals and penalizes
  observations outside them in proportion to the distance. It equals
  :math:`2 / \alpha` times the sum of the pinball losses of :math:`l` at
  level :math:`\alpha / 2` and of :math:`u` at level :math:`1 - \alpha / 2`.

Quantiles are equivariant under monotone transformations: if :math:`l_i`
and :math:`u_i` are quantiles of :math:`\log(1 + y_i)`, then
:math:`e^{l_i} - 1` and :math:`e^{u_i} - 1` are the same quantiles of
:math:`y_i`. Coverage is therefore identical on both scales, while width and
interval score are reported on the ``log1p`` scale (relative error, suffix
``_log``) and on the original scale, as for the point metrics of
:mod:`vgsales.evaluation.metrics`. Interval bounds are given on the original
scale and clipped at zero before scoring.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike

INTERVAL_METRICS: Final = (
    "coverage",
    "below",
    "above",
    "width_log",
    "interval_score_log",
    "width",
    "interval_score",
)


def _as_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must hold finite values")
    return array


def _check_level(value: float, name: str) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")
    return float(value)


def _as_interval(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = _as_array(y_true, "y_true")
    low = np.clip(_as_array(lower, "lower"), 0.0, None)
    high = np.clip(_as_array(upper, "upper"), 0.0, None)
    if not true.shape == low.shape == high.shape:
        raise ValueError("y_true, lower and upper must have the same length")
    if (true < 0).any():
        raise ValueError("observed sales must be non-negative")
    if (low > high).any():
        raise ValueError("lower bounds must not exceed upper bounds")
    return true, low, high


def pinball_loss(y_true: ArrayLike, y_quantile: ArrayLike, tau: float) -> float:
    """Mean pinball loss of predicted ``tau``-quantiles.

    Parameters
    ----------
    y_true : array-like of float
        Observed values.
    y_quantile : array-like of float
        Predicted quantiles on the same scale as ``y_true``.
    tau : float
        Quantile level in (0, 1).

    Returns
    -------
    float
        Mean of :math:`\\rho_\\tau(y - q)`.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty or not finite, or ``tau``
        lies outside (0, 1).
    """
    tau = _check_level(tau, "tau")
    true = _as_array(y_true, "y_true")
    quantile = _as_array(y_quantile, "y_quantile")
    if true.shape != quantile.shape:
        raise ValueError("y_true and y_quantile must have the same length")
    residual = true - quantile
    return float(np.mean(residual * (tau - (residual < 0))))


def _score(true: np.ndarray, low: np.ndarray, high: np.ndarray, alpha: float) -> float:
    penalty = (2.0 / alpha) * (
        (low - true) * (true < low) + (true - high) * (true > high)
    )
    return float(np.mean(high - low + penalty))


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Share of observations inside their closed interval ``[lower, upper]``."""
    true, low, high = _as_interval(y_true, lower, upper)
    return float(np.mean((true >= low) & (true <= high)))


def interval_score(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> float:
    """Mean interval score of central ``1 - alpha`` intervals.

    Parameters
    ----------
    y_true : array-like of float
        Observed sales on the original scale.
    lower, upper : array-like of float
        Interval bounds on the original scale.
    alpha : float
        Nominal miscoverage in (0, 1).

    Returns
    -------
    float
        Mean of :math:`S_\\alpha(l, u; y)` on the original scale.
    """
    alpha = _check_level(alpha, "alpha")
    return _score(*_as_interval(y_true, lower, upper), alpha)


def interval_metrics(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> dict[str, float]:
    """Compute coverage, width and interval score on both scales.

    Parameters
    ----------
    y_true : array-like of float
        Observed sales on the original scale, non-negative.
    lower, upper : array-like of float
        Bounds of the central ``1 - alpha`` intervals on the original scale.
    alpha : float
        Nominal miscoverage in (0, 1).

    Returns
    -------
    dict of str to float
        Values of :data:`INTERVAL_METRICS`, in that order.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty or not finite, sales are
        negative, a lower bound exceeds its upper bound, or ``alpha`` lies
        outside (0, 1).
    """
    alpha = _check_level(alpha, "alpha")
    true, low, high = _as_interval(y_true, lower, upper)
    log_true, log_low, log_high = np.log1p(true), np.log1p(low), np.log1p(high)
    return {
        "coverage": float(np.mean((true >= low) & (true <= high))),
        "below": float(np.mean(true < low)),
        "above": float(np.mean(true > high)),
        "width_log": float(np.mean(log_high - log_low)),
        "interval_score_log": _score(log_true, log_low, log_high, alpha),
        "width": float(np.mean(high - low)),
        "interval_score": _score(true, low, high, alpha),
    }
