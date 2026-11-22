r"""Reference models that every learned model has to beat.

All baselines take the pre-release feature frame and start with
:class:`vgsales.features.pipeline.LeakageGuard`. Under the default
``log1p`` target transform they are fitted on :math:`z = \log(1 + y)`.

- ``global_median``: the median of :math:`z` over the training rows. It
  ignores every input and sets the floor for skill.
- ``group_median``: the median of :math:`z` within each genre and console
  cell. A cell with fewer than ``min_count`` training rows backs off to the
  genre median, and an unseen or rare genre to the global median. This is
  the rule of thumb "titles of this kind on this platform usually sell this
  much".
- ``ridge``: ridge regression on the standardized output of
  :func:`vgsales.features.pipeline.make_preprocessor`, that is one-hot
  console and genre indicators, encoded publisher and developer, and the
  numeric pre-release features. It is the linear reference for the
  nonlinear models of later phases.

The median is equivariant under the monotone map :math:`z \mapsto e^z - 1`
when the group size is odd, so the median baselines predict the group median
of sales on the original scale up to the averaging of the two central values
in even-sized groups.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline

from vgsales.config import FeatureConfig
from vgsales.features.pipeline import LeakageGuard, make_preprocessor

BASELINE_NAMES: Final = ("global_median", "group_median", "ridge")

_MISSING: Final = "__missing__"
_SEP: Final = "\x1f"


class GroupMedianRegressor(RegressorMixin, BaseEstimator):
    """Median of the target within nested categorical cells, with back-off.

    Parameters
    ----------
    columns : sequence of str, default ("genre", "console")
        Grouping columns from the coarsest to the finest level. Prediction
        uses the finest level whose cell has at least ``min_count`` training
        rows, then drops the last column, down to the global median.
    min_count : int, default 5
        Minimum training rows for a cell median to be used.

    Attributes
    ----------
    global_median_ : float
        Median of the training target.
    medians_ : list of pandas.Series
        For each level from the finest to the coarsest, the cell medians of
        the cells with at least ``min_count`` rows, indexed by cell key.
    """

    def __init__(
        self,
        columns: Sequence[str] = ("genre", "console"),
        min_count: int = 5,
    ) -> None:
        self.columns = columns
        self.min_count = min_count

    def _levels(self) -> list[list[str]]:
        columns = list(self.columns)
        return [columns[:depth] for depth in range(len(columns), 0, -1)]

    def _keys(self, X: pd.DataFrame, columns: list[str]) -> pd.Series:
        parts = X.loc[:, columns].astype(object)
        parts = parts.where(parts.notna(), _MISSING).astype(str)
        key = parts[columns[0]]
        for column in columns[1:]:
            key = key + _SEP + parts[column]
        return key

    def _check_frame(self, X: object) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GroupMedianRegressor expects a pandas DataFrame")
        missing = sorted(set(self.columns) - set(X.columns))
        if missing:
            raise ValueError(f"X is missing grouping columns: {missing}")
        return X

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> GroupMedianRegressor:
        """Compute the global median and the cell medians of every level."""
        X = self._check_frame(X)
        if not self.columns:
            raise ValueError("columns must name at least one column")
        if self.min_count < 1:
            raise ValueError("min_count must be >= 1")
        target = pd.Series(np.asarray(y, dtype=np.float64), index=X.index)
        if len(target) == 0:
            raise ValueError("cannot fit on an empty frame")
        self.global_median_ = float(target.median())
        self.medians_ = []
        for columns in self._levels():
            grouped = target.groupby(self._keys(X, columns).to_numpy())
            stats = grouped.agg(["median", "size"])
            self.medians_.append(stats.loc[stats["size"] >= self.min_count, "median"])
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def backoff_level(self, X: pd.DataFrame) -> np.ndarray:
        """Return the number of grouping columns used for each row.

        Zero means the global median was used.
        """
        X = self._check_frame(X)
        depth = np.zeros(len(X), dtype=np.int64)
        for columns, medians in zip(self._levels(), self.medians_, strict=True):
            found = self._keys(X, columns).isin(medians.index).to_numpy()
            depth = np.where((depth == 0) & found, len(columns), depth)
        return depth

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the median of the finest supported cell of each row."""
        X = self._check_frame(X)
        pred = np.full(len(X), np.nan)
        for columns, medians in zip(self._levels(), self.medians_, strict=True):
            todo = np.isnan(pred)
            if not todo.any():
                break
            keys = self._keys(X.loc[todo], columns)
            pred[todo] = keys.map(medians).to_numpy(dtype=np.float64)
        return np.where(np.isnan(pred), self.global_median_, pred)


def make_baseline(
    name: str, config: FeatureConfig | None = None, **params: Any
) -> Pipeline:
    """Build an unfitted baseline pipeline.

    Parameters
    ----------
    name : {"global_median", "group_median", "ridge"}
        Baseline to build.
    config : FeatureConfig, optional
        Feature configuration for the ``ridge`` preprocessor.
    **params
        Keyword arguments of the final estimator:
        :class:`GroupMedianRegressor` for ``group_median`` and
        ``sklearn.linear_model.Ridge`` for ``ridge``. ``global_median``
        takes none.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline whose first step is a :class:`LeakageGuard`.

    Raises
    ------
    ValueError
        If ``name`` is unknown or parameters are given for ``global_median``.
    """
    if name == "global_median":
        if params:
            raise ValueError(f"global_median takes no parameters, got {params}")
        return make_pipeline(LeakageGuard(), DummyRegressor(strategy="median"))
    if name == "group_median":
        return make_pipeline(LeakageGuard(), GroupMedianRegressor(**params))
    if name == "ridge":
        return make_pipeline(make_preprocessor(config, scale=True), Ridge(**params))
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")
