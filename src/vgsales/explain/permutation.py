r"""Permutation importance on held-out rows and its agreement with SHAP.

Let :math:`L(f; X, y)` be a loss, such as the RMSLE, of the predictions of a
fitted model :math:`f` on held-out rows :math:`(X, y)`. For a group
:math:`G` of input columns and a random permutation :math:`\pi` of the rows,
let :math:`X^{\pi}_G` be :math:`X` with the columns in :math:`G` permuted
jointly by :math:`\pi`. The permutation importance of :math:`G` over
:math:`R` permutations is

.. math::

    I_G = \frac{1}{R} \sum_{r=1}^{R}
          \left[ L(f; X^{\pi_r}_G, y) - L(f; X, y) \right],

with the sign reversed for scores where greater is better (Breiman, 2001;
Fisher et al., 2019). It measures how much the held-out loss of the fitted
model depends on the information in :math:`G`. SHAP importance measures the
size of the attributions instead and needs no labels, so the two rankings
answer different questions and need not agree.

Columns that are linked by construction are permuted together, so that the
permuted rows remain coherent: the critic score with its missingness
indicator, and the release month with its quarter and holiday flag. Other
correlated columns are permuted separately, which creates rows outside the
joint distribution on which the model extrapolates (Hooker et al., 2021).

On a temporal holdout the permutation mixes values within the holdout years
only. A column that varies mainly between years, such as the release year,
is close to constant there, and its permutation importance is small by
construction even if the model relies on it across years.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import pandas as pd

from vgsales.evaluation.metrics import (
    GREATER_IS_BETTER,
    METRIC_NAMES,
    PRIMARY_METRIC,
    regression_metrics,
)
from vgsales.explain.tree_shap import ShapExplanation
from vgsales.features.prerelease import GROUP_COLUMN

LINKED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "critic_score": ("critic_score", "critic_missing"),
    "release_month": ("release_month", "release_quarter", "holiday_season"),
}


class Predictor(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def column_groups(
    columns: Sequence[str],
    linked: Mapping[str, Sequence[str]] = LINKED_COLUMNS,
) -> dict[str, tuple[str, ...]]:
    """Partition input columns into groups that are permuted jointly.

    Parameters
    ----------
    columns : sequence of str
        Input columns of the model.
    linked : mapping of str to sequence of str, optional
        Named sets of linked columns; members absent from ``columns`` are
        ignored.

    Returns
    -------
    dict of str to tuple of str
        Groups in the order in which their first member appears in
        ``columns``; unlinked columns form singleton groups named after
        themselves.

    Raises
    ------
    ValueError
        If a column belongs to more than one linked set.
    """
    owner: dict[str, str] = {}
    for name, members in linked.items():
        for column in members:
            if column in owner:
                raise ValueError(f"column {column!r} is linked more than once")
            owner[column] = name
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(owner.get(column, column), []).append(column)
    return {name: tuple(members) for name, members in groups.items()}


@dataclass(frozen=True)
class PermutationResult:
    """Held-out losses before and after permuting each group of columns.

    Attributes
    ----------
    metric : str
        Name of the score.
    baseline : float
        Score of the unpermuted rows.
    scores : pandas.DataFrame
        Score after each permutation, one row per group and one column per
        repeat.
    groups : dict of str to tuple of str
        Columns of each group.
    """

    metric: str
    baseline: float
    scores: pd.DataFrame
    groups: dict[str, tuple[str, ...]]

    def differences(self) -> pd.DataFrame:
        """Loss increase of each permutation; positive means the group helps."""
        if self.metric in GREATER_IS_BETTER:
            return self.baseline - self.scores
        return self.scores - self.baseline

    def importance(self) -> pd.DataFrame:
        """Mean and standard deviation of the loss increase, largest first.

        Returns
        -------
        pandas.DataFrame
            Indexed by group, with columns ``importance`` (mean over
            repeats), ``std`` (sample standard deviation, missing for one
            repeat) and ``columns`` (members of the group).
        """
        diffs = self.differences()
        table = pd.DataFrame(
            {
                "importance": diffs.mean(axis=1),
                "std": diffs.std(axis=1, ddof=1),
                "columns": [", ".join(self.groups[g]) for g in diffs.index],
            }
        )
        table.index.name = "group"
        return table.sort_values("importance", ascending=False, kind="stable")


def _score(model: Predictor, X: pd.DataFrame, y: np.ndarray, metric: str) -> float:
    return regression_metrics(y, model.predict(X), (metric,))[metric]


def permutation_importance(
    model: Predictor,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    groups: Mapping[str, Sequence[str]] | None = None,
    metric: str = PRIMARY_METRIC,
    n_repeats: int = 10,
    random_state: int = 0,
) -> PermutationResult:
    """Permutation importance of groups of input columns on held-out rows.

    Parameters
    ----------
    model : estimator
        Fitted model whose ``predict`` takes the pre-release feature frame
        and returns sales in millions of units.
    X : pandas.DataFrame
        Held-out feature frame.
    y : numpy.ndarray
        Observed sales of the rows of ``X``.
    groups : mapping of str to sequence of str, optional
        Columns permuted jointly, by group name. By default every column of
        ``X`` except the title key, grouped by :func:`column_groups`.
    metric : str, default "rmsle"
        Score from :data:`vgsales.evaluation.metrics.METRIC_NAMES`.
    n_repeats : int, default 10
        Permutations per group.
    random_state : int, default 0
        Seed of the permutations.

    Returns
    -------
    PermutationResult
        Baseline and permuted scores.

    Raises
    ------
    ValueError
        If the metric is unknown, ``n_repeats`` is not positive, ``X`` has
        fewer than two rows or a group names a missing column.
    """
    if metric not in METRIC_NAMES:
        raise ValueError(f"metric must be one of {METRIC_NAMES}, got {metric!r}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be positive, got {n_repeats}")
    if len(X) < 2:
        raise ValueError("permutation importance needs at least two rows")
    if groups is None:
        groups = column_groups([c for c in X.columns if c != GROUP_COLUMN])
    groups = {name: tuple(members) for name, members in groups.items()}
    missing = sorted({c for m in groups.values() for c in m} - set(X.columns))
    if missing:
        raise ValueError(f"columns not in X: {missing}")
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    baseline = _score(model, X, y, metric)
    scores = np.empty((len(groups), n_repeats))
    for i, members in enumerate(groups.values()):
        for r in range(n_repeats):
            order = rng.permutation(len(X))
            permuted = X.copy()
            for column in members:
                permuted[column] = X[column].iloc[order].set_axis(X.index)
            scores[i, r] = _score(model, permuted, y, metric)
    frame = pd.DataFrame(scores, index=pd.Index(list(groups), name="group"))
    return PermutationResult(metric, baseline, frame, groups)


def grouped_shap_importance(
    explanation: ShapExplanation, groups: Mapping[str, Sequence[str]]
) -> pd.Series:
    """Mean absolute SHAP value of each group of input columns.

    SHAP values are summed over the model columns of each input column and
    then over the input columns of each group before the absolute value is
    taken, which matches a joint permutation of the group.

    Parameters
    ----------
    explanation : ShapExplanation
        Output of :func:`vgsales.explain.tree_shap.explain_trees`.
    groups : mapping of str to sequence of str
        Input columns of each group; groups with no explained column are
        left out.

    Returns
    -------
    pandas.Series
        Importance by group, largest first.
    """
    values = explanation.grouped_values()
    importance = {}
    for name, members in groups.items():
        present = [column for column in members if column in values.columns]
        if present:
            importance[name] = float(values[present].sum(axis=1).abs().mean())
    series = pd.Series(importance, dtype=np.float64, name="mean_abs_shap")
    return series.sort_values(ascending=False, kind="stable")


def compare_rankings(permutation: pd.Series, shap: pd.Series) -> pd.DataFrame:
    """Place two importance vectors and their ranks side by side.

    Parameters
    ----------
    permutation, shap : pandas.Series
        Importance by group. Only groups present in both are compared.

    Returns
    -------
    pandas.DataFrame
        Columns ``permutation``, ``permutation_rank``, ``shap`` and
        ``shap_rank`` (1 is the most important, ties share the average rank),
        sorted by permutation importance.
    """
    common = [group for group in permutation.index if group in shap.index]
    table = pd.DataFrame(
        {
            "permutation": permutation.loc[common].astype(np.float64),
            "shap": shap.loc[common].astype(np.float64),
        }
    )
    table["permutation_rank"] = table["permutation"].rank(ascending=False)
    table["shap_rank"] = table["shap"].rank(ascending=False)
    table.index.name = "group"
    table = table.loc[:, ["permutation", "permutation_rank", "shap", "shap_rank"]]
    return table.sort_values("permutation", ascending=False, kind="stable")


def rank_correlation(table: pd.DataFrame) -> float:
    """Spearman rank correlation of the two rankings of :func:`compare_rankings`.

    Returns
    -------
    float
        Correlation in [-1, 1], or NaN with fewer than three groups or a
        constant ranking.
    """
    if len(table) < 3:
        return float("nan")
    ranks = table.loc[:, ["permutation_rank", "shap_rank"]]
    if (ranks.nunique() < 2).any():
        return float("nan")
    return float(ranks["permutation_rank"].corr(ranks["shap_rank"]))
