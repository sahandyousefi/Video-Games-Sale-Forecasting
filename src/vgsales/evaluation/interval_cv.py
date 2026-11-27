r"""Cross-validation of prediction intervals over release-year folds.

The runner mirrors :func:`vgsales.evaluation.cv.cross_validate`. For each
fold :math:`k`, a clone of the interval model is wrapped with the target
transform, fitted on the training rows and asked for the point prediction
and the central :math:`(1 - \alpha)` interval :math:`[l_i, u_i]` of every
validation row, on the original scale. Calibration steps of conformal
methods therefore use the training rows of the fold only.

Each fold is scored with :func:`vgsales.evaluation.intervals.interval_metrics`
and with the point metrics of the prediction. As for point forecasts, the
fold mean and the sample standard deviation across folds weight every
validation block equally, and the pooled value, computed once on all
out-of-fold intervals, weights every row equally. Pooled coverage is the
quantity the nominal level :math:`1 - \alpha` refers to under
exchangeability; the per-fold values show how it varies from one block of
release years to the next.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from vgsales.data.splits import ExpandingWindowSplit, Fold, describe_folds
from vgsales.evaluation.cv import (
    AGGREGATE_ROWS,
    FOLD_COLUMN,
    check_same_folds,
    check_target,
    fit_fold_model,
    resolve_folds,
)
from vgsales.evaluation.intervals import INTERVAL_METRICS, interval_metrics
from vgsales.evaluation.metrics import regression_metrics
from vgsales.models.quantile import predict_sales_interval

POINT_METRICS: Final = ("rmsle", "mae_log")
PRIMARY_INTERVAL_METRIC: Final = "interval_score_log"
PREDICTION_COLUMNS: Final = (FOLD_COLUMN, "y_true", "y_pred", "lower", "upper")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntervalCVResult:
    """Per-fold interval scores and out-of-fold intervals of one model.

    Attributes
    ----------
    per_fold : pandas.DataFrame
        One row per fold: the columns of
        :func:`vgsales.data.splits.describe_folds`, ``fit_seconds``, the
        interval metrics and the point metrics.
    predictions : pandas.DataFrame
        Out-of-fold predictions indexed by the index labels of ``X``, with
        the columns of :data:`PREDICTION_COLUMNS` on the original scale.
    alpha : float
        Nominal miscoverage of the intervals.
    metrics : tuple of str
        Names of the metric columns, interval metrics first.
    """

    per_fold: pd.DataFrame
    predictions: pd.DataFrame
    alpha: float
    metrics: tuple[str, ...]

    @property
    def point_metrics(self) -> tuple[str, ...]:
        """Names of the point metric columns."""
        return tuple(m for m in self.metrics if m not in INTERVAL_METRICS)

    def pooled(self) -> dict[str, float]:
        """Metrics of all out-of-fold intervals and predictions together."""
        frame = self.predictions
        scores = interval_metrics(
            frame["y_true"], frame["lower"], frame["upper"], self.alpha
        )
        if self.point_metrics:
            scores.update(
                regression_metrics(frame["y_true"], frame["y_pred"], self.point_metrics)
            )
        return scores

    def summary(self) -> pd.DataFrame:
        """Aggregate the per-fold scores.

        Returns
        -------
        pandas.DataFrame
            Rows ``mean``, ``std`` (``NaN`` for a single fold), ``min``,
            ``max`` and ``pooled``; one column per metric.
        """
        scores = self.per_fold.loc[:, list(self.metrics)]
        table = pd.DataFrame(
            {
                "mean": scores.mean(),
                "std": scores.std(ddof=1),
                "min": scores.min(),
                "max": scores.max(),
                "pooled": pd.Series(self.pooled()),
            }
        ).T
        return table.loc[list(AGGREGATE_ROWS), list(self.metrics)]


def check_alpha(alpha: float) -> float:
    """Return ``alpha`` as a float in (0, 1).

    Raises
    ------
    ValueError
        If ``alpha`` lies outside (0, 1).
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha}")
    return float(alpha)


def check_interval_model(estimator: BaseEstimator, alpha: float) -> None:
    """Check that ``estimator`` predicts intervals at the nominal level ``alpha``.

    Raises
    ------
    TypeError
        If the estimator has no ``predict_interval`` method.
    ValueError
        If the estimator's own ``alpha`` differs from ``alpha``.
    """
    if not hasattr(estimator, "predict_interval"):
        raise TypeError(f"{type(estimator).__name__} does not predict intervals")
    own = estimator.get_params(deep=False).get("alpha")
    if own is not None and not math.isclose(float(own), alpha):
        raise ValueError(
            f"{type(estimator).__name__} has alpha={own}, but the intervals are "
            f"scored at alpha={alpha}"
        )


def _score_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    point_metrics: tuple[str, ...],
) -> dict[str, float]:
    scores = interval_metrics(y_true, lower, upper, alpha)
    if point_metrics:
        scores.update(regression_metrics(y_true, y_pred, point_metrics))
    return scores


def cross_validate_intervals(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    cv: ExpandingWindowSplit | Sequence[Fold],
    *,
    alpha: float,
    groups: ArrayLike | None = None,
    target_transform: str = "log1p",
    point_metrics: tuple[str, ...] = POINT_METRICS,
) -> IntervalCVResult:
    """Evaluate the prediction intervals of a model on release-year folds.

    Parameters
    ----------
    estimator : estimator
        Unfitted interval model with ``predict_interval``, for example
        :class:`vgsales.models.quantile.QuantileIntervalRegressor`. It is
        cloned for every fold.
    X : pandas.DataFrame
        Pre-release feature frame.
    y : array-like of float
        Observed sales on the original scale, aligned by position with ``X``.
    cv : ExpandingWindowSplit or sequence of Fold
        Splitter, or precomputed folds with positional indices into ``X``.
    alpha : float
        Nominal miscoverage of the intervals, used by the interval score. It
        must match the estimator's ``alpha`` parameter when it has one.
    groups : array-like, optional
        Group labels passed to the splitter for purging.
    target_transform : {"log1p", "identity"}, default "log1p"
        Scale on which the estimator is fitted.
    point_metrics : tuple of str, optional
        Point metrics of the prediction, from
        :data:`vgsales.evaluation.metrics.METRIC_NAMES`.

    Returns
    -------
    IntervalCVResult
        Per-fold scores and out-of-fold intervals.

    Raises
    ------
    TypeError
        If ``X`` is not a DataFrame, ``cv`` has the wrong type or the
        estimator does not predict intervals.
    ValueError
        If ``alpha`` is invalid or differs from the estimator's, ``y`` is
        not aligned with ``X`` or the validation blocks overlap.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    alpha = check_alpha(alpha)
    check_interval_model(estimator, alpha)
    target = check_target(y, len(X))
    folds = resolve_folds(cv, X, groups)
    point_metrics = tuple(point_metrics)

    rows, predictions = [], []
    for k, fold in enumerate(folds):
        model, elapsed = fit_fold_model(estimator, X, target, fold, target_transform)
        X_val = X.iloc[fold.val_idx]
        y_true = target[fold.val_idx]
        y_pred = np.asarray(model.predict(X_val), dtype=np.float64)
        lower, upper = predict_sales_interval(model, X_val)
        scores = _score_fold(y_true, y_pred, lower, upper, alpha, point_metrics)
        rows.append({"fit_seconds": elapsed, **scores})
        predictions.append(
            pd.DataFrame(
                {
                    FOLD_COLUMN: k,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "lower": lower,
                    "upper": upper,
                },
                index=X.index[fold.val_idx],
            )
        )

    per_fold = pd.concat([describe_folds(folds), pd.DataFrame(rows)], axis=1)
    return IntervalCVResult(
        per_fold=per_fold,
        predictions=pd.concat(predictions),
        alpha=alpha,
        metrics=(*INTERVAL_METRICS, *point_metrics),
    )


def compare_intervals(
    estimators: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: ArrayLike,
    cv: ExpandingWindowSplit | Sequence[Fold],
    *,
    alpha: float,
    groups: ArrayLike | None = None,
    target_transform: str = "log1p",
    point_metrics: tuple[str, ...] = POINT_METRICS,
) -> dict[str, IntervalCVResult]:
    """Cross-validate several interval models on one set of folds.

    Parameters
    ----------
    estimators : mapping of str to estimator
        Unfitted interval models keyed by method name.
    X, y, cv, alpha, groups, target_transform, point_metrics
        As in :func:`cross_validate_intervals`.

    Returns
    -------
    dict of str to IntervalCVResult
        Results in the order of ``estimators``.

    Raises
    ------
    ValueError
        If ``estimators`` is empty.
    """
    if not estimators:
        raise ValueError("estimators must not be empty")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    folds = resolve_folds(cv, X, groups)
    results = {}
    for name, estimator in estimators.items():
        logger.info("cross-validating the intervals of %s", name)
        results[name] = cross_validate_intervals(
            estimator,
            X,
            y,
            folds,
            alpha=alpha,
            target_transform=target_transform,
            point_metrics=point_metrics,
        )
    return results


def interval_comparison_table(results: Mapping[str, IntervalCVResult]) -> pd.DataFrame:
    """Tabulate fold mean, fold standard deviation and pooled interval scores.

    Parameters
    ----------
    results : mapping of str to IntervalCVResult
        Results keyed by method name, computed on the same folds at the same
        nominal level.

    Returns
    -------
    pandas.DataFrame
        One row per method with columns ``<metric>_mean``, ``<metric>_std``
        and ``<metric>_pooled``, sorted by the pooled interval score on the
        ``log1p`` scale.

    Raises
    ------
    ValueError
        If the results differ in their folds or nominal level.
    """
    check_same_folds(results)
    levels = {result.alpha for result in results.values()}
    if len(levels) > 1:
        raise ValueError(f"results were scored at different levels: {sorted(levels)}")
    rows = {}
    for name, result in results.items():
        summary = result.summary()
        rows[name] = {
            f"{metric}_{stat}": summary.loc[stat, metric]
            for metric in result.metrics
            for stat in ("mean", "std", "pooled")
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "method"
    return table.sort_values(f"{PRIMARY_INTERVAL_METRIC}_pooled")
