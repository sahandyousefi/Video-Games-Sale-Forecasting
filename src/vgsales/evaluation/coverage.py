r"""Coverage of prediction intervals by release year and by genre.

Pooled coverage close to the nominal level :math:`1 - \alpha` can hide
groups that are over- or under-covered, for example release years after a
market shift or genres whose sales spread differently from the rest. The
functions below score the out-of-fold intervals of
:mod:`vgsales.evaluation.interval_cv` within groups of rows.

For a group :math:`g` with :math:`n_g` rows of which :math:`c_g` are covered,
the coverage :math:`\hat p_g = c_g / n_g` is reported with the Wilson score
interval (Wilson, 1927)

.. math::

    \frac{\hat p + z^2 / (2 n)}{1 + z^2 / n} \pm \frac{z}{1 + z^2 / n}
    \sqrt{\frac{\hat p (1 - \hat p)}{n} + \frac{z^2}{4 n^2}},
    \qquad z = \Phi^{-1}\left(\frac{1 + \gamma}{2}\right),

at confidence level :math:`\gamma`, which stays inside :math:`[0, 1]` and
keeps close to its nominal level for small groups and coverage near one. The
interval treats the rows of a group as independent Bernoulli trials. Rows of
one release year share market conditions and ports of one title are
correlated, so the band understates the sampling uncertainty.

The reliability figure plots the coverage of every method by release year
and by genre, with these bands, against the nominal level.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import norm

from vgsales.evaluation.interval_cv import IntervalCVResult
from vgsales.evaluation.intervals import interval_metrics

if TYPE_CHECKING:
    from matplotlib.figure import Figure

GROUP_COLUMNS: Final = (
    "n",
    "covered",
    "coverage",
    "coverage_low",
    "coverage_high",
    "below",
    "above",
    "width_log",
    "interval_score_log",
)
METHOD_COLUMN: Final = "method"

# First three categorical slots of the reference palette, one marker each so
# that the methods stay distinguishable without colour.
_COLORS: Final = ("#257de0", "#f47230", "#00b894")
_MARKERS: Final = ("o", "s", "^")
_TEXT: Final = "#191919"
_MUTED: Final = "#666666"
_GRID: Final = "#e6e6e6"


def wilson_interval(
    covered: ArrayLike, n: ArrayLike, confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval of binomial proportions.

    Parameters
    ----------
    covered : array-like of int
        Number of successes of each group.
    n : array-like of int
        Number of trials of each group, positive.
    confidence : float, default 0.95
        Confidence level in (0, 1).

    Returns
    -------
    lower, upper : numpy.ndarray
        Bounds of each proportion.

    Raises
    ------
    ValueError
        If a count is invalid or ``confidence`` lies outside (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    k = np.asarray(covered, dtype=np.float64)
    trials = np.asarray(n, dtype=np.float64)
    if k.shape != trials.shape:
        raise ValueError("covered and n must have the same shape")
    if (trials <= 0).any() or (k < 0).any() or (k > trials).any():
        raise ValueError("counts must satisfy 0 <= covered <= n and n > 0")
    z = norm.ppf(0.5 + confidence / 2)
    p = k / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    half = z / denominator * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    # The interval contains p exactly; min and max only absorb rounding error.
    lower = np.clip(np.minimum(center - half, p), 0.0, 1.0)
    upper = np.clip(np.maximum(center + half, p), 0.0, 1.0)
    return lower, upper


def _aligned_labels(predictions: pd.DataFrame, labels: pd.Series) -> pd.Series:
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series indexed like the rows of X")
    aligned = labels.reindex(predictions.index)
    if aligned.isna().any():
        raise ValueError(
            f"labels are missing for {int(aligned.isna().sum())} predicted rows"
        )
    return aligned


def coverage_by(
    predictions: pd.DataFrame,
    labels: pd.Series,
    alpha: float,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Interval scores of the out-of-fold predictions within each group.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Out-of-fold intervals with columns ``y_true``, ``lower`` and
        ``upper``, as in :attr:`IntervalCVResult.predictions`.
    labels : pandas.Series
        Group label of each row, indexed like the feature frame, for example
        ``X["release_year"]``.
    alpha : float
        Nominal miscoverage of the intervals.
    confidence : float, default 0.95
        Level of the Wilson interval of the coverage.

    Returns
    -------
    pandas.DataFrame
        One row per group, sorted by label and indexed by it, with the
        columns of :data:`GROUP_COLUMNS`.

    Raises
    ------
    ValueError
        If a predicted row has no label.
    TypeError
        If ``labels`` is not a Series.
    """
    aligned = _aligned_labels(predictions, labels)
    rows = {}
    for label, frame in predictions.groupby(aligned.to_numpy(), sort=True):
        scores = interval_metrics(
            frame["y_true"], frame["lower"], frame["upper"], alpha
        )
        n = len(frame)
        rows[label] = {
            "n": n,
            "covered": round(scores["coverage"] * n),
            **{name: scores[name] for name in GROUP_COLUMNS[2:] if name in scores},
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    low, high = wilson_interval(table["covered"], table["n"], confidence)
    table["coverage_low"], table["coverage_high"] = low, high
    table.index.name = labels.name
    return table.loc[:, list(GROUP_COLUMNS)]


def coverage_table(
    results: Mapping[str, IntervalCVResult],
    labels: pd.Series,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Coverage by group for several interval methods.

    Parameters
    ----------
    results : mapping of str to IntervalCVResult
        Cross-validated intervals keyed by method name.
    labels : pandas.Series
        Group label of each row, indexed like the feature frame.
    confidence : float, default 0.95
        Level of the Wilson interval of the coverage.

    Returns
    -------
    pandas.DataFrame
        Long table with the columns ``method``, the name of ``labels`` and
        :data:`GROUP_COLUMNS`, in the order of ``results``.
    """
    name = labels.name if labels.name is not None else "group"
    frames = []
    for method, result in results.items():
        table = coverage_by(
            result.predictions,
            labels.rename(name),
            result.alpha,
            confidence=confidence,
        ).reset_index()
        table.insert(0, METHOD_COLUMN, method)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def _style(ax, grid_axis: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelcolor=_TEXT, labelsize=9)
    ax.grid(axis=grid_axis, color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)


def _methods(table: pd.DataFrame) -> list[str]:
    methods = list(dict.fromkeys(table[METHOD_COLUMN]))
    if len(methods) > len(_COLORS):
        raise ValueError(f"the figure shows at most {len(_COLORS)} methods")
    return methods


def plot_coverage_reliability(
    by_year: pd.DataFrame,
    by_genre: pd.DataFrame,
    alpha: float,
    *,
    min_count: int = 10,
    path: str | Path | None = None,
) -> Figure:
    """Plot coverage by release year and by genre against the nominal level.

    Parameters
    ----------
    by_year, by_genre : pandas.DataFrame
        Outputs of :func:`coverage_table` with ``release_year`` and
        ``genre`` labels.
    alpha : float
        Nominal miscoverage of the intervals.
    min_count : int, default 10
        Groups with fewer rows are left out, since their coverage is too
        noisy to read.
    path : str or Path, optional
        File to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the release-year panel on the left and the genre panel
        on the right.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If more than three methods are given.
    """
    try:
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator
    except ImportError as error:
        raise ImportError(
            'the reliability figure needs matplotlib: pip install "vgsales[viz]"'
        ) from error

    nominal = 1.0 - alpha
    methods = _methods(by_year)
    years = by_year[by_year["n"] >= min_count]
    genres = by_genre[by_genre["n"] >= min_count]

    fig = Figure(figsize=(11.0, 4.6), layout="constrained")
    year_ax, genre_ax = fig.subplots(1, 2, width_ratios=(1.6, 1.0))

    for color, marker, method in zip(_COLORS, _MARKERS, methods, strict=False):
        rows = years[years[METHOD_COLUMN] == method].sort_values("release_year")
        x = rows["release_year"].to_numpy()
        year_ax.fill_between(
            x,
            rows["coverage_low"],
            rows["coverage_high"],
            color=color,
            alpha=0.15,
            lw=0,
        )
        year_ax.plot(
            x, rows["coverage"], color=color, lw=2, marker=marker, ms=5, label=method
        )
    year_ax.axhline(nominal, color=_MUTED, lw=1, ls="--")
    year_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    year_ax.set_xlabel("Release year")
    year_ax.set_ylabel("Empirical coverage")
    year_ax.set_title("Coverage by release year", loc="left", fontsize=11)
    _style(year_ax, "y")

    order = sorted(genres["genre"].astype(str).unique())
    position = {genre: i for i, genre in enumerate(order)}
    offsets = np.linspace(-0.2, 0.2, len(methods)) if len(methods) > 1 else [0.0]
    for color, marker, offset, method in zip(
        _COLORS, _MARKERS, offsets, methods, strict=False
    ):
        rows = genres[genres[METHOD_COLUMN] == method]
        y = rows["genre"].astype(str).map(position).to_numpy() + offset
        error = np.vstack(
            [
                rows["coverage"] - rows["coverage_low"],
                rows["coverage_high"] - rows["coverage"],
            ]
        )
        genre_ax.errorbar(
            rows["coverage"],
            y,
            xerr=error,
            fmt=marker,
            color=color,
            ms=5,
            elinewidth=1.5,
            capsize=0,
        )
    genre_ax.axvline(nominal, color=_MUTED, lw=1, ls="--")
    genre_ax.set_yticks(range(len(order)), order)
    genre_ax.invert_yaxis()
    genre_ax.set_xlabel("Empirical coverage")
    genre_ax.set_title("Coverage by genre", loc="left", fontsize=11)
    _style(genre_ax, "x")

    low = min(years["coverage_low"].min(), genres["coverage_low"].min(), nominal)
    lower_limit = max(0.0, np.floor((low - 0.05) * 10) / 10)
    year_ax.set_ylim(lower_limit, 1.01)
    genre_ax.set_xlim(lower_limit, 1.01)
    handles, labels = year_ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=_MUTED, lw=1, ls="--"))
    labels.append(f"nominal {nominal:.0%}")
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=_TEXT,
    )
    fig.suptitle(
        f"Out-of-fold coverage of central {nominal:.0%} intervals; bands and "
        f"bars are Wilson intervals; groups with at least {min_count} rows",
        x=0.01,
        ha="left",
        fontsize=10,
        color=_MUTED,
    )
    if path is not None:
        fig.savefig(path, dpi=150, facecolor="white")
    return fig
