r"""Local explanations and dependence plots built on TreeSHAP values.

Local explanation. For a row :math:`x_i` and an input column :math:`c` with
model columns :math:`J_c`, the grouped attribution is
:math:`\psi_c(x_i) = \sum_{j \in J_c} \phi_j(x_i)`, and local accuracy gives

.. math::

    f(x_i) = \phi_0 + \sum_c \psi_c(x_i).

The input columns are ranked by :math:`|\psi_c(x_i)|`; the first ``top`` are
listed and the rest are summed into one term, so the decomposition stays
exact. On the :math:`\log(1 + y)` scale the forecast is
:math:`\hat y_i = e^{f(x_i)} - 1`, and each term multiplies
:math:`1 + \hat y_i` by :math:`e^{\psi_c(x_i)}`.

Dependence plot. For a model column :math:`j`, the points
:math:`(x_{ij}, \phi_j(x_i))` show how the attribution changes with the
value of the column. Under an additive model the points would lie on one
curve; vertical spread at a given value comes from interactions with other
columns (Lundberg et al., 2020), which colouring by a second column can
reveal. Nominal inputs passed to a model as category codes have no order
and are not plotted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from vgsales.data.temporal import RELEASE_YEAR
from vgsales.explain.tree_shap import (
    _BAR,
    _MUTED,
    _RAMP,
    _TEXT,
    ShapExplanation,
    _colour_values,
    _figure_class,
    _save,
    _style,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_NEGATIVE: Final = "#ec4647"
_LABEL_COLUMNS: Final = ("title_key", "console", RELEASE_YEAR)
_DISCRETE_LEVELS: Final = 12


@dataclass(frozen=True)
class LocalExplanation:
    """Additive decomposition of the prediction for one row.

    Attributes
    ----------
    label : str
        Description of the row, such as title, console and release year.
    base_value : float
        Expected prediction :math:`\\phi_0` on the modelling scale.
    contributions : pandas.DataFrame
        Indexed by input column, largest absolute attribution first, with
        columns ``value`` (the input value as text) and ``shap`` (grouped
        SHAP value). The last row sums the remaining columns when some were
        left out.
    transform : str, default "log1p"
        Target transform of the modelling scale.
    """

    label: str
    base_value: float
    contributions: pd.DataFrame
    transform: str = "log1p"

    @property
    def prediction(self) -> float:
        """Prediction on the modelling scale."""
        return self.base_value + float(self.contributions["shap"].sum())

    def predicted_sales(self) -> float:
        """Prediction in millions of units."""
        if self.transform == "log1p":
            return float(np.expm1(self.prediction))
        return self.prediction


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float | np.floating) and np.isnan(value)):
        return "missing"
    if isinstance(value, float | np.floating):
        return f"{value:.3g}"
    return str(value)


def row_label(row: pd.Series) -> str:
    """Describe a row of the feature frame by title, console and release year.

    Parameters
    ----------
    row : pandas.Series
        Row of the pre-release feature frame.

    Returns
    -------
    str
        For example ``"super mario 64 (N64, 1996)"``; the row name when the
        frame has no title column.
    """
    if "title_key" not in row.index:
        return str(row.name)
    details = [_format_value(row[c]) for c in _LABEL_COLUMNS[1:] if c in row.index]
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{row['title_key']}{suffix}"


def explain_row(
    explanation: ShapExplanation,
    X: pd.DataFrame,
    index: Any,
    *,
    top: int = 8,
    transform: str = "log1p",
    label: str | None = None,
) -> LocalExplanation:
    """Decompose the prediction for one explained row by input column.

    Parameters
    ----------
    explanation : ShapExplanation
        Output of :func:`vgsales.explain.tree_shap.explain_trees`.
    X : pandas.DataFrame
        Pre-release feature frame that holds the explained rows; it supplies
        the input values and the label.
    index : hashable
        Index label of the row, which must be among the explained rows.
    top : int, default 8
        Number of input columns listed; the rest are summed.
    transform : str, default "log1p"
        Target transform of the modelling scale.
    label : str, optional
        Description of the row; built by :func:`row_label` when omitted.

    Returns
    -------
    LocalExplanation
        Decomposition whose terms add up to the prediction.

    Raises
    ------
    ValueError
        If the row was not explained or ``top`` is not positive.
    """
    if top < 1:
        raise ValueError(f"top must be positive, got {top}")
    if index not in explanation.values.index:
        raise ValueError(f"row {index!r} is not among the explained rows")
    grouped = explanation.grouped_values().loc[index]
    order = grouped.abs().sort_values(ascending=False, kind="stable").index
    shown, rest = list(order[:top]), list(order[top:])
    inputs = X.loc[index]
    values = []
    for column in shown:
        if column in inputs.index:
            values.append(_format_value(inputs[column]))
        else:
            values.append(_format_value(explanation.data.loc[index, column]))
    shap_values = [float(grouped[column]) for column in shown]
    names = list(shown)
    if rest:
        names.append(f"{len(rest)} other columns")
        values.append("")
        shap_values.append(float(grouped[rest].sum()))
    contributions = pd.DataFrame(
        {"value": values, "shap": shap_values}, index=pd.Index(names, name="feature")
    )
    return LocalExplanation(
        label=label or row_label(inputs),
        base_value=explanation.base_value,
        contributions=contributions,
        transform=transform,
    )


def _scale_label(transform: str) -> str:
    return "log(1 + sales)" if transform == "log1p" else "sales (millions of units)"


def plot_local_explanation(
    local: LocalExplanation,
    *,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Waterfall chart of a local explanation.

    The bottom bar starts at the expected prediction and each bar adds one
    term, so the top bar ends at the prediction for the row. Terms that
    raise the forecast are blue and terms that lower it are red; each bar is
    labelled with its signed value.

    Parameters
    ----------
    local : LocalExplanation
        Output of :func:`explain_row`.
    title : str, optional
        Axes title; by default the row label and the predicted sales.
    path : str or Path, optional
        File to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    Figure = _figure_class()
    terms = local.contributions
    n_terms = len(terms)
    fig = Figure(figsize=(7.5, 0.42 * n_terms + 1.8), layout="constrained")
    ax = fig.subplots()
    start = local.base_value
    span = float(terms["shap"].abs().sum()) or 1.0
    # Bars are drawn bottom-up so that the cumulative sum ends at the top row.
    for position, value in enumerate(terms["shap"].to_numpy()[::-1]):
        value = float(value)
        colour = _BAR if value >= 0 else _NEGATIVE
        ax.barh(position, value, left=start, height=0.6, color=colour)
        end = start + value
        offset = 0.01 * span
        ax.text(
            max(start, end) + offset,
            position,
            f"{value:+.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color=_TEXT,
        )
        start = end
    labels = [
        f"{name} = {row['value']}" if row["value"] else str(name)
        for name, row in terms.iterrows()
    ]
    ax.set_yticks(range(n_terms), list(reversed(labels)))
    ax.axvline(local.base_value, color=_MUTED, lw=0.8, ls="--")
    ax.axvline(local.prediction, color=_TEXT, lw=0.8)
    ax.set_ylim(-0.7, n_terms - 0.3)
    scale = _scale_label(local.transform)
    ax.set_xlabel(
        f"{scale}: expected {local.base_value:.3f} (dashed), "
        f"prediction {local.prediction:.3f} (solid)"
    )
    ax.set_title(
        title
        or f"{local.label}: predicted {local.predicted_sales():.3f} million units",
        loc="left",
        fontsize=11,
    )
    _style(ax)
    ax.margins(x=0.12)
    _save(fig, path)
    return fig


def _check_ordered(explanation: ShapExplanation, column: str) -> None:
    if column not in explanation.values.columns:
        raise ValueError(f"unknown model column {column!r}")
    source = explanation.sources.get(column, column)
    if column == source and source in explanation.categorical:
        raise ValueError(
            f"{column!r} holds category codes of a nominal input and has no order"
        )


def _jitter(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    levels = np.unique(x[np.isfinite(x)])
    if len(levels) < 2 or len(levels) > _DISCRETE_LEVELS:
        return x
    width = 0.2 * float(np.min(np.diff(levels)))
    return x + rng.uniform(-width, width, size=len(x))


def plot_shap_dependence(
    explanation: ShapExplanation,
    column: str,
    *,
    color_by: str | None = None,
    max_points: int = 2000,
    random_state: int = 0,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Dependence plot of the SHAP value of one model column on its value.

    Rows with a missing value of ``column`` are not drawn; their number is
    given in the axis label. Columns with at most 12 distinct values are
    jittered horizontally.

    Parameters
    ----------
    explanation : ShapExplanation
        Output of :func:`vgsales.explain.tree_shap.explain_trees`.
    column : str
        Model column on the horizontal axis.
    color_by : str, optional
        Second model column whose rank colours the points, from light (low)
        to dark (high); missing values are grey.
    max_points : int, default 2000
        Rows drawn at most; larger sets are subsampled.
    random_state : int, default 0
        Seed of the jitter and of the subsample.
    title : str, optional
        Axes title.
    path : str or Path, optional
        File to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        If a column is unknown or holds category codes of a nominal input.
    """
    _check_ordered(explanation, column)
    if color_by is not None:
        _check_ordered(explanation, color_by)
    Figure = _figure_class()
    from matplotlib.colors import LinearSegmentedColormap

    rng = np.random.default_rng(random_state)
    values, data = explanation.values, explanation.data
    if len(values) > max_points:
        keep = rng.choice(len(values), size=max_points, replace=False)
        values, data = values.iloc[keep], data.iloc[keep]
    x = data[column].to_numpy(dtype=np.float64)
    y = values[column].to_numpy(dtype=np.float64)
    observed = np.isfinite(x)
    n_missing = int((~observed).sum())
    x_plot = _jitter(x, rng)

    fig = Figure(figsize=(6.5, 4.2), layout="constrained")
    ax = fig.subplots()
    if color_by is None:
        ax.scatter(x_plot[observed], y[observed], s=8, color=_BAR, lw=0, alpha=0.7)
    else:
        cmap = LinearSegmentedColormap.from_list("shap_values", list(_RAMP))
        colour = _colour_values(data[color_by])
        grey = observed & ~np.isfinite(colour)
        coloured = observed & np.isfinite(colour)
        ax.scatter(x_plot[grey], y[grey], s=8, color="#9c9c9c", lw=0, alpha=0.6)
        points = ax.scatter(
            x_plot[coloured],
            y[coloured],
            s=8,
            c=colour[coloured],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            lw=0,
            alpha=0.8,
        )
        bar = fig.colorbar(points, ax=ax, pad=0.01, aspect=40)
        bar.set_ticks([0.0, 1.0], labels=["low", "high"])
        bar.set_label(f"{color_by} (rank)", color=_TEXT, fontsize=9)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=_MUTED, labelcolor=_TEXT, labelsize=8, length=0)
    ax.axhline(0.0, color=_MUTED, lw=0.8)
    note = f" ({n_missing} missing not shown)" if n_missing else ""
    ax.set_xlabel(f"{column}{note}")
    ax.set_ylabel("SHAP value, log(1 + sales) scale", color=_TEXT, fontsize=9)
    ax.set_title(title or f"Dependence on {column}", loc="left", fontsize=11)
    _style(ax)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    _save(fig, path)
    return fig


def ordered_columns(explanation: ShapExplanation, n: int) -> list[str]:
    """Return the ``n`` most important model columns that carry an order.

    Category codes and one-hot indicators of nominal inputs are left out,
    so encodings such as ``publisher_te`` and numeric columns remain.

    Parameters
    ----------
    explanation : ShapExplanation
        Output of :func:`vgsales.explain.tree_shap.explain_trees`.
    n : int
        Number of columns.

    Returns
    -------
    list of str
        Model columns, most important first.
    """
    nominal = set(explanation.categorical)
    columns = [
        column
        for column in explanation.importance().index
        if explanation.sources.get(column, column) not in nominal
    ]
    return columns[: max(n, 0)]


def spread_rows(predictions: pd.Series, k: int) -> list[Any]:
    """Return ``k`` rows at evenly spaced ranks of the predictions.

    The first row has the highest prediction and the last the lowest, so
    three rows give the maximum, the median and the minimum.

    Parameters
    ----------
    predictions : pandas.Series
        Predictions indexed by row.
    k : int
        Number of rows; fewer are returned when the series is shorter.

    Returns
    -------
    list
        Index labels, highest prediction first.
    """
    if k < 1 or predictions.empty:
        return []
    ordered = predictions.sort_values(ascending=False, kind="stable")
    if k == 1:
        return [ordered.index[0]]
    ranks = np.linspace(0, len(ordered) - 1, num=min(k, len(ordered)))
    positions = np.unique(np.round(ranks).astype(int))
    return list(ordered.index[positions])
