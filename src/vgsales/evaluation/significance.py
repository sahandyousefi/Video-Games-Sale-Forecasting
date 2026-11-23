r"""Paired comparison of models on shared cross-validation folds.

Let models :math:`A` and :math:`B` be evaluated on the same :math:`K` folds,
with scores :math:`m_k^A` and :math:`m_k^B` of one metric on fold :math:`k`.
The paired differences are :math:`d_k = m_k^A - m_k^B`. Folds with
:math:`d_k = 0` are dropped (Wilcoxon, 1945), leaving :math:`n` differences.
Their absolute values are ranked, with tied values receiving the mean of
the ranks they span, and the signed-rank statistic is

.. math::

    W^+ = \sum_{k: d_k > 0} R_k ,

the sum of the ranks :math:`R_k` of the positive differences. Under the null
hypothesis that the distribution of :math:`d_k` is symmetric about zero,
each of the :math:`2^n` sign assignments is equally likely and
:math:`E[W^+] = n(n + 1)/4`. The two-sided p-value is the probability of a
deviation :math:`|W^+ - E[W^+]|` at least as large as the observed one. It
is computed exactly by enumerating the sign assignments for
:math:`n \le` :data:`EXACT_MAX_N`, which remains exact when ranks are tied,
and otherwise from the normal approximation with variance

.. math::

    \operatorname{Var}(W^+) = \frac{n(n + 1)(2n + 1)}{24}
        - \frac{\sum_j (t_j^3 - t_j)}{48},

where :math:`t_j` is the size of the :math:`j`-th group of tied ranks.

With few folds the test has little power. The smallest attainable two-sided
p-value is :math:`2^{1-n}`, which is 0.0625 for :math:`n = 5`: five folds
cannot reject the null hypothesis at the 5% level whatever the scores. The
folds of an expanding window also share training rows, so the differences
are not independent, and the test is a descriptive check on the consistency
of a ranking across validation years rather than a formal inference on
generalization error (Nadeau and Bengio, 2003; Demsar, 2006).

When every model is compared with one reference model, the p-values are
adjusted for multiple testing with the step-down procedure of Holm (1979).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import rankdata

from vgsales.evaluation.cv import CVResult, check_same_folds, comparison_table
from vgsales.evaluation.metrics import GREATER_IS_BETTER, PRIMARY_METRIC

EXACT_MAX_N: Final = 20

PAIRED_COLUMNS: Final = (
    "reference",
    "n_folds",
    "n_better",
    "median_diff",
    "statistic",
    "p_value",
    "p_holm",
    "p_min",
)


@dataclass(frozen=True)
class SignedRankResult:
    """Outcome of a Wilcoxon signed-rank test.

    Attributes
    ----------
    n : int
        Number of nonzero differences.
    statistic : float
        Sum of the ranks of the positive differences, :math:`W^+`.
    p_value : float
        Two-sided p-value; 1.0 when every difference is zero.
    p_min : float
        Smallest two-sided p-value attainable with ``n`` differences.
    exact : bool
        Whether the p-value was computed by enumeration.
    """

    n: int
    statistic: float
    p_value: float
    p_min: float
    exact: bool


def _exact_p(ranks: np.ndarray, statistic: float) -> float:
    n = ranks.size
    signs = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    totals = signs @ ranks
    center = n * (n + 1) / 4.0
    observed = abs(statistic - center)
    # Rank sums are multiples of 0.5, so a small tolerance is exact.
    return float(np.mean(np.abs(totals - center) >= observed - 1e-9))


def _normal_p(ranks: np.ndarray, statistic: float) -> float:
    n = ranks.size
    _, counts = np.unique(ranks, return_counts=True)
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    variance -= float(np.sum(counts**3 - counts)) / 48.0
    if variance <= 0:
        return 1.0
    z = abs(statistic - n * (n + 1) / 4.0) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2.0))


def signed_rank_test(differences: ArrayLike) -> SignedRankResult:
    """Two-sided Wilcoxon signed-rank test of paired differences.

    Parameters
    ----------
    differences : array-like of float
        Paired differences :math:`d_k`, one per fold.

    Returns
    -------
    SignedRankResult
        Statistic, p-value and the smallest attainable p-value.

    Raises
    ------
    ValueError
        If ``differences`` is empty, not one-dimensional or not finite.
    """
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional array")
    if not np.isfinite(d).all():
        raise ValueError("differences must be finite")
    d = d[d != 0]
    n = int(d.size)
    if n == 0:
        return SignedRankResult(0, 0.0, 1.0, 1.0, True)
    ranks = rankdata(np.abs(d), method="average")
    statistic = float(ranks[d > 0].sum())
    exact = n <= EXACT_MAX_N
    p_value = _exact_p(ranks, statistic) if exact else _normal_p(ranks, statistic)
    return SignedRankResult(
        n=n,
        statistic=statistic,
        p_value=min(1.0, p_value),
        p_min=min(1.0, 2.0 ** (1 - n)),
        exact=exact,
    )


def holm_adjust(p_values: ArrayLike) -> np.ndarray:
    """Adjust p-values with the step-down procedure of Holm (1979).

    Parameters
    ----------
    p_values : array-like of float
        Unadjusted p-values of :math:`m` hypotheses.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values in the input order: the running maximum of
        :math:`(m - i + 1)\\, p_{(i)}` over the sorted p-values, capped at 1.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    m = p.size
    scaled = (m - np.arange(m)) * p[order]
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted
    return out


def paired_comparison(
    results: Mapping[str, CVResult],
    metric: str = PRIMARY_METRIC,
    reference: str | None = None,
) -> pd.DataFrame:
    """Compare every model with a reference model fold by fold.

    Parameters
    ----------
    results : mapping of str to CVResult
        Cross-validation results on the same folds, at least two models.
    metric : str, default "rmsle"
        Metric whose per-fold scores are paired.
    reference : str, optional
        Model to compare against. Defaults to the model with the best pooled
        score of ``metric``.

    Returns
    -------
    pandas.DataFrame
        One row per model other than the reference, indexed by model name,
        with columns ``reference``, ``n_folds``, ``n_better`` (folds on which
        the model beats the reference), ``median_diff`` (median of
        :math:`m_k^{model} - m_k^{reference}`), ``statistic`` (:math:`W^+`),
        ``p_value``, ``p_holm`` and ``p_min``, sorted by ``p_value``.

    Raises
    ------
    ValueError
        If fewer than two models are given, the folds differ, the metric was
        not computed or ``reference`` is unknown.
    """
    if len(results) < 2:
        raise ValueError("a paired comparison needs at least two models")
    check_same_folds(results)
    if any(metric not in result.metrics for result in results.values()):
        raise ValueError(f"metric {metric!r} was not computed for every model")
    if reference is None:
        pooled = comparison_table(results)[f"{metric}_pooled"]
        best = pooled.idxmax() if metric in GREATER_IS_BETTER else pooled.idxmin()
        reference = str(best)
    if reference not in results:
        raise ValueError(f"unknown reference model {reference!r}")

    base = results[reference].per_fold[metric].to_numpy(dtype=np.float64)
    sign = -1.0 if metric in GREATER_IS_BETTER else 1.0
    rows = {}
    for name, result in results.items():
        if name == reference:
            continue
        diff = result.per_fold[metric].to_numpy(dtype=np.float64) - base
        test = signed_rank_test(diff)
        rows[name] = {
            "reference": reference,
            "n_folds": int(diff.size),
            "n_better": int(np.sum(sign * diff < 0)),
            "median_diff": float(np.median(diff)),
            "statistic": test.statistic,
            "p_value": test.p_value,
            "p_min": test.p_min,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["p_holm"] = holm_adjust(table["p_value"].to_numpy())
    table.index.name = "model"
    return table.loc[:, list(PAIRED_COLUMNS)].sort_values("p_value", kind="stable")
