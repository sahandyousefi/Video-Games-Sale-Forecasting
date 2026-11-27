r"""Cross-validation runner over release-year folds.

For each fold :math:`k` the estimator is cloned, wrapped with the target
transform, fitted on the training rows and used to predict the validation
rows. Every learned statistic, including those of the preprocessing pipeline,
is therefore estimated from the training rows of the fold only.

Two aggregates are reported for each metric :math:`m`. The fold mean
:math:`\bar m = K^{-1} \sum_k m_k` with the sample standard deviation across
folds weights every validation year equally. The pooled value :math:`m_\cup`
is computed once on the out-of-fold predictions of all folds concatenated,
which weights every row equally and so gives more weight to years with more
releases. Under the expanding-window scheme the validation blocks are
disjoint, so each row contributes at most one out-of-fold prediction.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone

from vgsales.data.splits import ExpandingWindowSplit, Fold, describe_folds
from vgsales.evaluation.metrics import (
    METRIC_NAMES,
    PRIMARY_METRIC,
    regression_metrics,
)
from vgsales.models.target import with_target_transform

FOLD_COLUMN = "fold"
AGGREGATE_ROWS = ("mean", "std", "min", "max", "pooled")
FOLD_KEYS = (
    "fold",
    "train_start",
    "train_end",
    "val_start",
    "val_end",
    "n_train",
    "n_val",
    "n_purged",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVResult:
    """Per-fold scores and out-of-fold predictions of one estimator.

    Attributes
    ----------
    per_fold : pandas.DataFrame
        One row per fold: the columns of
        :func:`vgsales.data.splits.describe_folds`, ``fit_seconds`` and one
        column per metric.
    predictions : pandas.DataFrame
        Out-of-fold predictions indexed by the index labels of ``X``, with
        columns ``fold``, ``y_true`` and ``y_pred`` on the original scale.
        ``y_pred`` is stored as returned by the estimator; the metrics clip
        it at zero.
    metrics : tuple of str
        Names of the metric columns.
    """

    per_fold: pd.DataFrame
    predictions: pd.DataFrame
    metrics: tuple[str, ...]

    def summary(self) -> pd.DataFrame:
        """Aggregate the per-fold scores.

        Returns
        -------
        pandas.DataFrame
            Rows ``mean``, ``std`` (sample standard deviation across folds,
            ``NaN`` for a single fold), ``min``, ``max`` and ``pooled``
            (metrics of all out-of-fold predictions together); one column per
            metric.
        """
        scores = self.per_fold.loc[:, list(self.metrics)]
        pooled = regression_metrics(
            self.predictions["y_true"], self.predictions["y_pred"], self.metrics
        )
        table = pd.DataFrame(
            {
                "mean": scores.mean(),
                "std": scores.std(ddof=1),
                "min": scores.min(),
                "max": scores.max(),
                "pooled": pd.Series(pooled),
            }
        ).T
        return table.loc[list(AGGREGATE_ROWS), list(self.metrics)]


def check_target(y: ArrayLike, n: int) -> np.ndarray:
    """Return ``y`` as a float array of ``n`` finite, non-negative sales.

    Raises
    ------
    ValueError
        If ``y`` has the wrong shape or invalid values.
    """
    values = np.asarray(y, dtype=np.float64)
    if values.shape != (n,):
        raise ValueError(f"y must be one-dimensional with {n} values")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("y must hold finite, non-negative sales")
    return values


def resolve_folds(
    cv: ExpandingWindowSplit | Sequence[Fold],
    X: pd.DataFrame,
    groups: ArrayLike | None = None,
) -> list[Fold]:
    """Return the folds of ``cv`` for ``X`` and check that they are disjoint.

    Parameters
    ----------
    cv : ExpandingWindowSplit or sequence of Fold
        Splitter, or precomputed folds with positional indices into ``X``.
    X : pandas.DataFrame
        Feature frame.
    groups : array-like, optional
        Group labels passed to the splitter for purging.

    Raises
    ------
    TypeError
        If ``cv`` has the wrong type.
    ValueError
        If ``groups`` is given with precomputed folds, a fold indexes beyond
        ``X`` or two validation blocks overlap.
    """
    if isinstance(cv, ExpandingWindowSplit):
        folds = cv.folds(X, groups)
    else:
        folds = list(cv)
        if not folds or not all(isinstance(f, Fold) for f in folds):
            raise TypeError("cv must be an ExpandingWindowSplit or a sequence of Fold")
        if groups is not None:
            raise ValueError("groups are only used with an ExpandingWindowSplit")
    seen = np.zeros(len(X), dtype=bool)
    for k, fold in enumerate(folds):
        if fold.val_idx.size and fold.val_idx.max() >= len(X):
            raise ValueError(f"fold {k} indexes beyond the {len(X)} rows of X")
        if seen[fold.val_idx].any():
            raise ValueError(f"validation rows of fold {k} overlap an earlier fold")
        seen[fold.val_idx] = True
    return folds


def fit_fold_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    target: np.ndarray,
    fold: Fold,
    target_transform: str = "log1p",
) -> tuple[BaseEstimator, float]:
    """Fit a clone of ``estimator`` on the training rows of one fold.

    Parameters
    ----------
    estimator : estimator
        Unfitted regressor; it is cloned and wrapped with the target
        transform.
    X : pandas.DataFrame
        Feature frame.
    target : numpy.ndarray
        Observed sales on the original scale, aligned with ``X``.
    fold : Fold
        Positional training and validation indices.
    target_transform : {"log1p", "identity"}, default "log1p"
        Scale on which the estimator is fitted.

    Returns
    -------
    model : estimator
        Fitted estimator that predicts sales on the original scale.
    fit_seconds : float
        Wall-clock time of the fit.
    """
    model = with_target_transform(clone(estimator), target_transform)
    start = time.perf_counter()
    model.fit(X.iloc[fold.train_idx], target[fold.train_idx])
    return model, time.perf_counter() - start


def fit_fold(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    target: np.ndarray,
    fold: Fold,
    target_transform: str = "log1p",
) -> tuple[np.ndarray, float]:
    """Fit a clone of ``estimator`` on one fold and predict its validation rows.

    Parameters are as for :func:`fit_fold_model`.

    Returns
    -------
    y_pred : numpy.ndarray
        Predictions for the validation rows on the original scale.
    fit_seconds : float
        Wall-clock time of the fit.
    """
    model, elapsed = fit_fold_model(estimator, X, target, fold, target_transform)
    y_pred = np.asarray(model.predict(X.iloc[fold.val_idx]), dtype=np.float64)
    return y_pred, elapsed


def cross_validate(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    cv: ExpandingWindowSplit | Sequence[Fold],
    *,
    groups: ArrayLike | None = None,
    target_transform: str = "log1p",
    metrics: tuple[str, ...] = METRIC_NAMES,
) -> CVResult:
    """Evaluate an estimator on release-year folds.

    Parameters
    ----------
    estimator : estimator
        Unfitted scikit-learn regressor, typically a pipeline that starts
        with :func:`vgsales.features.pipeline.make_preprocessor`. It is
        cloned for every fold.
    X : pandas.DataFrame
        Feature frame, for example the output of
        :func:`vgsales.features.prerelease.build_prerelease_features`.
    y : array-like of float
        Observed sales on the original scale, aligned by position with ``X``.
    cv : ExpandingWindowSplit or sequence of Fold
        Splitter, or precomputed folds with positional indices into ``X``.
    groups : array-like, optional
        Group labels passed to the splitter for purging.
    target_transform : {"log1p", "identity"}, default "log1p"
        Scale on which the estimator is fitted.
    metrics : tuple of str, optional
        Metric names from :data:`vgsales.evaluation.metrics.METRIC_NAMES`.

    Returns
    -------
    CVResult
        Per-fold metrics and out-of-fold predictions.

    Raises
    ------
    ValueError
        If ``y`` is not aligned with ``X`` or holds invalid sales, or the
        validation blocks overlap.
    TypeError
        If ``X`` is not a DataFrame or ``cv`` has the wrong type.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    target = check_target(y, len(X))
    folds = resolve_folds(cv, X, groups)

    rows, predictions = [], []
    for k, fold in enumerate(folds):
        y_pred, elapsed = fit_fold(estimator, X, target, fold, target_transform)
        y_true = target[fold.val_idx]
        rows.append(
            {"fit_seconds": elapsed, **regression_metrics(y_true, y_pred, metrics)}
        )
        predictions.append(
            pd.DataFrame(
                {FOLD_COLUMN: k, "y_true": y_true, "y_pred": y_pred},
                index=X.index[fold.val_idx],
            )
        )

    per_fold = pd.concat([describe_folds(folds), pd.DataFrame(rows)], axis=1)
    return CVResult(
        per_fold=per_fold,
        predictions=pd.concat(predictions),
        metrics=tuple(metrics),
    )


def compare_models(
    estimators: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: ArrayLike,
    cv: ExpandingWindowSplit | Sequence[Fold],
    *,
    groups: ArrayLike | None = None,
    target_transform: str = "log1p",
    metrics: tuple[str, ...] = METRIC_NAMES,
) -> dict[str, CVResult]:
    """Cross-validate several estimators on one set of folds.

    The folds are computed once and passed to :func:`cross_validate` for
    every estimator, so all models are trained and validated on the same
    rows and their per-fold scores are paired.

    Parameters
    ----------
    estimators : mapping of str to estimator
        Unfitted estimators keyed by model name.
    X, y, cv, groups, target_transform, metrics
        As in :func:`cross_validate`.

    Returns
    -------
    dict of str to CVResult
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
        logger.info("cross-validating %s on %d folds", name, len(folds))
        results[name] = cross_validate(
            estimator,
            X,
            y,
            folds,
            target_transform=target_transform,
            metrics=metrics,
        )
    return results


def check_same_folds(results: Mapping[str, CVResult]) -> None:
    """Check that all results were computed on the same folds.

    Two results share their folds when the year ranges and sizes of every
    fold agree and every out-of-fold prediction comes from the same fold for
    the same row.

    Raises
    ------
    ValueError
        If two results differ in their folds.
    """
    items = list(results.items())
    if not items:
        return
    first_name, first = items[0]
    reference = first.per_fold.loc[:, list(FOLD_KEYS)].reset_index(drop=True)
    for name, result in items[1:]:
        folds = result.per_fold.loc[:, list(FOLD_KEYS)].reset_index(drop=True)
        same = folds.equals(reference) and result.predictions[FOLD_COLUMN].equals(
            first.predictions[FOLD_COLUMN]
        )
        if not same:
            raise ValueError(
                f"{name} and {first_name} were not evaluated on the same folds"
            )


def comparison_table(results: Mapping[str, CVResult]) -> pd.DataFrame:
    """Tabulate fold mean, fold standard deviation and pooled scores.

    Parameters
    ----------
    results : mapping of str to CVResult
        Cross-validation results keyed by model name, computed on the same
        folds.

    Returns
    -------
    pandas.DataFrame
        One row per model with columns ``<metric>_mean``, ``<metric>_std``
        and ``<metric>_pooled``, sorted by the pooled primary metric when it
        was computed.

    Raises
    ------
    ValueError
        If the results were computed on different folds.
    """
    check_same_folds(results)
    rows = {}
    for name, result in results.items():
        summary = result.summary()
        rows[name] = {
            f"{metric}_{stat}": summary.loc[stat, metric]
            for metric in result.metrics
            for stat in ("mean", "std", "pooled")
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "model"
    key = f"{PRIMARY_METRIC}_pooled"
    return table.sort_values(key) if key in table.columns else table
