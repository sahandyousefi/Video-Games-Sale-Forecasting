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

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone

from vgsales.data.splits import ExpandingWindowSplit, Fold, describe_folds
from vgsales.evaluation.metrics import METRIC_NAMES, regression_metrics
from vgsales.models.target import with_target_transform

FOLD_COLUMN = "fold"
AGGREGATE_ROWS = ("mean", "std", "min", "max", "pooled")


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


def _as_target(y: ArrayLike, n: int) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64)
    if values.shape != (n,):
        raise ValueError(f"y must be one-dimensional with {n} values")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("y must hold finite, non-negative sales")
    return values


def _resolve_folds(
    cv: ExpandingWindowSplit | Sequence[Fold],
    X: pd.DataFrame,
    groups: ArrayLike | None,
) -> list[Fold]:
    if isinstance(cv, ExpandingWindowSplit):
        return cv.folds(X, groups)
    folds = list(cv)
    if not folds or not all(isinstance(f, Fold) for f in folds):
        raise TypeError("cv must be an ExpandingWindowSplit or a sequence of Fold")
    if groups is not None:
        raise ValueError("groups are only used with an ExpandingWindowSplit")
    return folds


def _check_disjoint(folds: list[Fold], n: int) -> None:
    seen = np.zeros(n, dtype=bool)
    for k, fold in enumerate(folds):
        if fold.val_idx.size and fold.val_idx.max() >= n:
            raise ValueError(f"fold {k} indexes beyond the {n} rows of X")
        if seen[fold.val_idx].any():
            raise ValueError(f"validation rows of fold {k} overlap an earlier fold")
        seen[fold.val_idx] = True


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
    target = _as_target(y, len(X))
    folds = _resolve_folds(cv, X, groups)
    _check_disjoint(folds, len(X))

    rows, predictions = [], []
    for k, fold in enumerate(folds):
        model = with_target_transform(clone(estimator), target_transform)
        start = time.perf_counter()
        model.fit(X.iloc[fold.train_idx], target[fold.train_idx])
        elapsed = time.perf_counter() - start
        y_pred = np.asarray(model.predict(X.iloc[fold.val_idx]), dtype=np.float64)
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
