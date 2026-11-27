r"""SHAP values and global importance of the gradient-boosting models.

Let :math:`f` be a fitted boosting model on the modelling scale
:math:`z = \log(1 + y)` and :math:`x \in \mathbb{R}^p` a row of its input
matrix after preprocessing. The SHAP value :math:`\phi_j(x)` of column
:math:`j` (Lundberg and Lee, 2017) is the Shapley value of :math:`j` in the
game whose value of a coalition :math:`S` is the expected prediction given
the features in :math:`S`. The values satisfy local accuracy,

.. math::

    f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j(x),

where :math:`\phi_0` is the expected prediction. TreeSHAP computes them
exactly for tree ensembles in polynomial time (Lundberg et al., 2020). It
is run here without background data, in its path-dependent form: the
expectation over the features outside :math:`S` follows the share of
training rows that went down each branch of every tree. Correlated columns,
such as the frequency and the temporal target encoding of a publisher,
therefore share credit according to the splits the trees happened to choose,
and the values describe the model, not causal effects of the features.

Because :math:`f` predicts :math:`\log(1 + y)`, the attributions are
additive on that scale: a value :math:`\phi_j(x)` multiplies the forecast of
:math:`1 + y` by :math:`e^{\phi_j(x)}`.

The global importance of column :math:`j` over :math:`n` rows is
:math:`I_j = n^{-1} \sum_i |\phi_j(x_i)|`. One input column can yield several
model columns (one indicator per console under one-hot encoding, a frequency
and a target encoding per publisher). Since SHAP values are additive, the
attribution of an input column is the sum of the values of its model columns,
and grouped importance takes the mean absolute value of that sum.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.utils.validation import check_is_fitted

from vgsales.config import FeatureConfig
from vgsales.data.temporal import RELEASE_YEAR
from vgsales.models.boosting import TemporalBoostingRegressor

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_BAR: Final = "#257de0"
_TEXT: Final = "#191919"
_MUTED: Final = "#666666"
_GRID: Final = "#e6e6e6"
_NEUTRAL: Final = "#9c9c9c"
# Single-hue blue ramp, light to dark, for feature values from low to high.
_RAMP: Final = ("#cfe2f8", "#257de0", "#0b3a75")


@dataclass(frozen=True)
class ShapExplanation:
    """SHAP values of a boosting model on a set of rows.

    Attributes
    ----------
    values : pandas.DataFrame
        SHAP values on the modelling scale, one row per explained row and one
        column per model column after preprocessing.
    data : pandas.DataFrame
        Model columns of the explained rows, coerced to numbers; categories
        passed as strings become missing.
    base_value : float
        Expected prediction :math:`\\phi_0` on the modelling scale.
    sources : dict of str to str
        Input column from which each model column is derived.
    categorical : tuple of str
        Input columns that are nominal, whose model columns carry codes or
        indicators rather than ordered values.
    """

    values: pd.DataFrame
    data: pd.DataFrame
    base_value: float
    sources: dict[str, str]
    categorical: tuple[str, ...] = ()

    def predictions(self) -> np.ndarray:
        """Modelling-scale predictions reconstructed from the SHAP values."""
        return self.base_value + self.values.to_numpy().sum(axis=1)

    def importance(self) -> pd.Series:
        """Mean absolute SHAP value of each model column, largest first."""
        importance = self.values.abs().mean().sort_values(ascending=False)
        return importance.rename("mean_abs_shap")

    def grouped_values(self) -> pd.DataFrame:
        """SHAP values summed over the model columns of each input column."""
        groups = [self.sources[column] for column in self.values.columns]
        summed = self.values.T.groupby(groups, sort=False).sum().T
        return summed

    def grouped_importance(self) -> pd.Series:
        """Mean absolute grouped SHAP value of each input column, largest first."""
        importance = self.grouped_values().abs().mean().sort_values(ascending=False)
        importance.index.name = None
        return importance.rename("mean_abs_shap")


def source_columns(names: Iterable[str], config: FeatureConfig) -> dict[str, str]:
    """Map model column names to the input columns they are derived from.

    A name equal to an input column maps to itself. Otherwise it maps to the
    longest input column ``c`` such that the name starts with ``c + "_"``,
    which covers one-hot indicators (``console_PS4``) and encodings
    (``publisher_freq``, ``publisher_te``). Names without a match map to
    themselves.

    Parameters
    ----------
    names : iterable of str
        Output names of the preprocessing pipeline.
    config : FeatureConfig
        Feature configuration of the pipeline.

    Returns
    -------
    dict of str to str
        Input column of each name.
    """
    inputs = list(
        dict.fromkeys(
            (
                *config.categorical,
                *config.high_cardinality,
                RELEASE_YEAR,
                *config.numeric,
            )
        )
    )
    mapping = {}
    for name in names:
        if name in inputs:
            mapping[name] = name
            continue
        prefixes = [column for column in inputs if name.startswith(f"{column}_")]
        mapping[name] = max(prefixes, key=len) if prefixes else name
    return mapping


def _boosting_model(model: BaseEstimator) -> TemporalBoostingRegressor:
    inner = model
    if isinstance(model, TransformedTargetRegressor):
        check_is_fitted(model, "regressor_")
        inner = model.regressor_
    if not isinstance(inner, TemporalBoostingRegressor):
        raise TypeError(
            "TreeSHAP explains the gradient-boosting models, got "
            f"{type(inner).__name__}"
        )
    check_is_fitted(inner, "booster_")
    return inner


def _shap_module() -> Any:
    try:
        import shap
    except ImportError as error:
        raise ImportError(
            'SHAP values need the shap package: pip install "vgsales[explain]"'
        ) from error
    return shap


def explain_trees(
    model: BaseEstimator,
    X: pd.DataFrame,
    *,
    max_rows: int | None = 2000,
    random_state: int = 0,
) -> ShapExplanation:
    """Compute TreeSHAP values of a fitted boosting model.

    Parameters
    ----------
    model : estimator
        Fitted :class:`vgsales.models.boosting.TemporalBoostingRegressor`,
        bare or wrapped by :func:`vgsales.models.target.with_target_transform`.
    X : pandas.DataFrame
        Pre-release feature frame of the rows to explain.
    max_rows : int or None, default 2000
        Rows beyond this number are subsampled without replacement; None
        explains every row.
    random_state : int, default 0
        Seed of the subsample.

    Returns
    -------
    ShapExplanation
        SHAP values indexed like the explained rows of ``X``.

    Raises
    ------
    TypeError
        If the model is not a fitted boosting model or ``X`` is not a frame.
    ImportError
        If shap is not installed.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    inner = _boosting_model(model)
    shap = _shap_module()
    rows = X
    if max_rows is not None and len(X) > max_rows:
        rows = X.sample(n=max_rows, random_state=random_state)
    names = [str(name) for name in inner.preprocessor_.get_feature_names_out()]
    features = inner.preprocessor_.transform(rows)
    if isinstance(features, pd.DataFrame):
        features = features.set_axis(names, axis=1).set_axis(rows.index, axis=0)
    else:
        features = pd.DataFrame(features, columns=names, index=rows.index)
    explainer = shap.TreeExplainer(inner.booster_)
    values = np.asarray(explainer.shap_values(features), dtype=np.float64)
    config = inner.config or FeatureConfig()
    return ShapExplanation(
        values=pd.DataFrame(values, columns=names, index=rows.index),
        data=features.apply(pd.to_numeric, errors="coerce"),
        base_value=float(np.ravel(explainer.expected_value)[0]),
        sources=source_columns(names, config),
        categorical=tuple(config.categorical),
    )


def _figure_class() -> Any:
    try:
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError(
            'SHAP figures need matplotlib: pip install "vgsales[explain]"'
        ) from error
    return Figure


def _style(ax: Any) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelcolor=_TEXT, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)


def _save(fig: Any, path: str | Path | None) -> None:
    if path is not None:
        fig.savefig(path, dpi=150, facecolor="white")


def plot_shap_importance(
    explanation: ShapExplanation,
    *,
    grouped: bool = True,
    top: int = 15,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Horizontal bar chart of the global SHAP importance.

    Parameters
    ----------
    explanation : ShapExplanation
        Output of :func:`explain_trees`.
    grouped : bool, default True
        Rank input columns (:meth:`ShapExplanation.grouped_importance`)
        instead of model columns.
    top : int, default 15
        Number of columns shown, largest at the top.
    title : str, optional
        Axes title.
    path : str or Path, optional
        File to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    Figure = _figure_class()
    importance = (
        explanation.grouped_importance() if grouped else explanation.importance()
    )
    shown = importance.head(top).iloc[::-1]
    fig = Figure(figsize=(7.0, 0.32 * len(shown) + 1.4), layout="constrained")
    ax = fig.subplots()
    ax.barh(shown.index.astype(str), shown.to_numpy(), height=0.6, color=_BAR)
    ax.set_xlabel("Mean |SHAP value| on the log(1 + sales) scale")
    ax.set_title(
        title or ("Global importance by input column" if grouped else "Importance"),
        loc="left",
        fontsize=11,
    )
    _style(ax)
    _save(fig, path)
    return fig


def _colour_values(column: pd.Series) -> np.ndarray:
    """Scale the ranks of a column to [0, 1]; missing values stay missing.

    Constant columns get no colour, and indicators map to 0 and 1.
    """
    ranks = column.rank(method="average").to_numpy(dtype=np.float64)
    low, high = np.nanmin(ranks, initial=np.inf), np.nanmax(ranks, initial=-np.inf)
    if not np.isfinite(low) or high <= low:
        return np.full(len(column), np.nan)
    return (ranks - low) / (high - low)


def plot_shap_summary(
    explanation: ShapExplanation,
    *,
    top: int = 15,
    max_points: int = 2000,
    random_state: int = 0,
    title: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Summary plot of the SHAP values of the most important model columns.

    Each point is one row; its horizontal position is the SHAP value and its
    colour the rank of the column's value among the explained rows, from
    light (low) to dark (high), so that indicators are light when absent and
    dark when present. Integer codes of nominal inputs, whose order carries
    no meaning, and missing values are drawn in grey.

    Parameters
    ----------
    explanation : ShapExplanation
        Output of :func:`explain_trees`.
    top : int, default 15
        Number of model columns shown, most important at the top.
    max_points : int, default 2000
        Rows drawn per column at most; larger sets are subsampled.
    random_state : int, default 0
        Seed of the vertical jitter and of the subsample.
    title : str, optional
        Axes title.
    path : str or Path, optional
        File to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    Figure = _figure_class()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("shap_values", list(_RAMP))
    rng = np.random.default_rng(random_state)
    columns = list(explanation.importance().head(top).index)
    values, data = explanation.values, explanation.data
    if len(values) > max_points:
        keep = rng.choice(len(values), size=max_points, replace=False)
        values, data = values.iloc[keep], data.iloc[keep]

    fig = Figure(figsize=(7.5, 0.38 * len(columns) + 1.6), layout="constrained")
    ax = fig.subplots()
    nominal = set(explanation.categorical)
    for position, column in enumerate(reversed(columns)):
        shap_values = values[column].to_numpy()
        y = position + rng.uniform(-0.28, 0.28, size=len(shap_values))
        colour = _colour_values(data[column])
        if column in nominal:
            colour = np.full(len(shap_values), np.nan)
        missing = ~np.isfinite(colour)
        ax.scatter(
            shap_values[missing], y[missing], s=6, color=_NEUTRAL, lw=0, alpha=0.6
        )
        points = ax.scatter(
            shap_values[~missing],
            y[~missing],
            s=6,
            c=colour[~missing],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            lw=0,
            alpha=0.8,
        )
    ax.axvline(0.0, color=_MUTED, lw=0.8)
    ax.set_yticks(range(len(columns)), list(reversed(columns)))
    ax.set_xlabel("SHAP value on the log(1 + sales) scale")
    ax.set_title(title or "SHAP values by row", loc="left", fontsize=11)
    _style(ax)
    bar = fig.colorbar(points, ax=ax, pad=0.01, aspect=40)
    bar.set_ticks([0.0, 1.0], labels=["low", "high"])
    bar.set_label("Feature value (rank)", color=_TEXT, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=_MUTED, labelcolor=_TEXT, labelsize=8, length=0)
    _save(fig, path)
    return fig


def importance_table(explanations: Mapping[str, ShapExplanation]) -> pd.DataFrame:
    """Grouped importance of several models side by side.

    Parameters
    ----------
    explanations : mapping of str to ShapExplanation
        Explanations keyed by model name.

    Returns
    -------
    pandas.DataFrame
        One row per input column and one column per model, sorted by the
        importance under the first model.
    """
    table = pd.DataFrame(
        {name: item.grouped_importance() for name, item in explanations.items()}
    )
    if table.empty:
        return table
    table.index.name = "feature"
    return table.sort_values(table.columns[0], ascending=False)
