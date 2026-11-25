r"""Rolling-origin evaluation of the monthly series forecasters.

Rolling-origin evaluation (Tashman, 2000) repeats a forecast from several
origins. For an origin :math:`o`, every forecaster is fitted on the months
up to and including :math:`o` and forecasts months :math:`o + 1, \dots, o +
H`, which are then compared with the observed values. The origins
:math:`o_1 < \dots < o_R` are spaced ``step`` months apart and the last one
leaves exactly :math:`H` observed months, so later origins see longer
histories, as they would in use. No forecaster sees a month after its
origin, and each one is refitted at every origin.

For segment :math:`k`, origin :math:`o` and step :math:`h`, let
:math:`r = \log(1 + \hat S_{k,o+h}) - \log(1 + S_{k,o+h})`. The scores pool
these errors over segments, origins and steps:

- ``rmsle``: :math:`\sqrt{\operatorname{mean}\, r^2}`;
- ``mae_log``: :math:`\operatorname{mean} |r|`;
- ``mase``: :math:`\operatorname{mean} |r| / Q_{k,o}`, the mean absolute
  scaled error of Hyndman and Koehler (2006) with the in-sample seasonal
  naive scale

  .. math::

      Q_{k,o} = \frac{1}{o - m + 1} \sum_{t=m}^{o}
          \big|\log(1 + S_{k,t}) - \log(1 + S_{k,t-m})\big|,

  computed on the history of the origin, so that a value below one beats
  the in-sample seasonal naive forecast; segments with :math:`Q_{k,o} = 0`
  are left out;
- ``mae``: mean absolute error in units.

The skill of a model against a reference is
:math:`1 - \text{rmsle}_{\text{model}} / \text{rmsle}_{\text{ref}}`.
Forecast errors of successive origins are correlated through shared
history and overlapping targets, so :func:`paired_origin_tests` pairs whole
origins and should be read as a check on the consistency of a ranking.

The series must end before the censoring window
(:func:`vgsales.data.series.uncensored_end`); otherwise the last origins
are scored against months whose sales are still accumulating.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from vgsales.evaluation.significance import holm_adjust, signed_rank_test
from vgsales.models.forecasting import SEASON_LENGTH, SeriesForecaster

SERIES_METRICS: Final = ("rmsle", "mae_log", "mase", "mae")
ERROR_COLUMNS: Final = (
    "model",
    "origin",
    "segment",
    "step",
    "actual",
    "forecast",
    "scale",
)


def rolling_origins(
    index: pd.DatetimeIndex,
    horizon: int,
    n_origins: int,
    step: int = 12,
    min_history: int = 36,
) -> pd.DatetimeIndex:
    """Return evaluation origins spaced ``step`` months apart.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Monthly grid of the series.
    horizon : int
        Forecast months :math:`H`; the last origin is ``horizon`` months
        before the end of ``index``.
    n_origins : int
        Number of origins :math:`R`.
    step : int, default 12
        Months between consecutive origins.
    min_history : int, default 36
        Minimum number of months up to and including the first origin.

    Returns
    -------
    pandas.DatetimeIndex
        Origins in increasing order.
    """
    if horizon < 1 or n_origins < 1 or step < 1 or min_history < 1:
        raise ValueError("horizon, n_origins, step and min_history must be >= 1")
    last = len(index) - 1 - horizon
    positions = last - step * np.arange(n_origins)[::-1]
    if positions[0] < min_history - 1:
        raise ValueError(
            f"{len(index)} months are too few for {n_origins} origins "
            f"{step} months apart with {min_history} months of history and "
            f"a {horizon}-month horizon"
        )
    return pd.DatetimeIndex(index[positions])


def seasonal_scale(history: pd.DataFrame, season: int = SEASON_LENGTH) -> np.ndarray:
    """Return :math:`Q_{k,o}` for every segment, ``NaN`` where it is zero."""
    z = np.log1p(history.to_numpy(dtype=np.float64))
    if len(z) <= season:
        raise ValueError(f"the history needs more than {season} months")
    scale = np.abs(z[season:] - z[:-season]).mean(axis=0)
    return np.where(scale > 0, scale, np.nan)


def _origin_errors(
    name: str,
    model: SeriesForecaster,
    wide: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    season: int,
) -> pd.DataFrame:
    history = wide.loc[:origin]
    forecast = model.fit(history).forecast()
    if len(forecast) < horizon:
        raise ValueError(
            f"{name} forecasts {len(forecast)} months, fewer than horizon={horizon}"
        )
    forecast = forecast.iloc[:horizon]
    actual = wide.loc[forecast.index, forecast.columns]
    if len(actual) != horizon:
        raise ValueError(f"{name} returned forecast months outside the series")
    scale = pd.Series(seasonal_scale(history, season), index=wide.columns)
    n_segments = forecast.shape[1]
    return pd.DataFrame(
        {
            "model": name,
            "origin": origin,
            "segment": np.tile(forecast.columns.to_numpy(dtype=object), horizon),
            "step": np.repeat(np.arange(1, horizon + 1), n_segments),
            "actual": actual.to_numpy(dtype=np.float64).ravel(),
            "forecast": forecast.to_numpy(dtype=np.float64).ravel(),
            "scale": np.tile(scale[forecast.columns].to_numpy(), horizon),
        }
    )


def _scores(errors: pd.DataFrame) -> pd.Series:
    log_error = np.log1p(errors["forecast"]) - np.log1p(errors["actual"])
    scaled = (log_error.abs() / errors["scale"]).dropna()
    return pd.Series(
        {
            "rmsle": float(np.sqrt(np.mean(log_error**2))),
            "mae_log": float(log_error.abs().mean()),
            "mase": float(scaled.mean()) if len(scaled) else float("nan"),
            "mae": float((errors["forecast"] - errors["actual"]).abs().mean()),
            "n": len(errors),
        }
    )


@dataclass(frozen=True)
class RollingResult:
    """Forecast errors of every model, origin, segment and step.

    Attributes
    ----------
    errors : pandas.DataFrame
        Long frame with the columns of :data:`ERROR_COLUMNS`; ``scale`` is
        :math:`Q_{k,o}`.
    origins : pandas.DatetimeIndex
        Evaluation origins.
    horizon : int
        Forecast months :math:`H`.
    """

    errors: pd.DataFrame
    origins: pd.DatetimeIndex
    horizon: int

    @property
    def models(self) -> list[str]:
        """Model names in evaluation order."""
        return list(pd.unique(self.errors["model"]))

    def scores(self, by: Sequence[str] = ()) -> pd.DataFrame:
        """Return the scores of each model, optionally per ``by`` group.

        Parameters
        ----------
        by : sequence of str, default ()
            Further grouping columns, for example ``("step",)`` or
            ``("origin",)``.

        Returns
        -------
        pandas.DataFrame
            One row per model (and group) with the columns of
            :data:`SERIES_METRICS` and the number of errors ``n``.
        """
        keys = ["model", *by]
        rows = {
            key: _scores(group) for key, group in self.errors.groupby(keys, sort=False)
        }
        table = pd.DataFrame(rows).T
        table.index.names = keys
        table["n"] = table["n"].astype(np.int64)
        return table

    def summary(self, reference: str | None = None) -> pd.DataFrame:
        """Return pooled scores, with skill against ``reference`` if given."""
        table = self.scores()
        table.index = table.index.get_level_values("model")
        if reference is not None:
            if reference not in table.index:
                raise ValueError(f"unknown reference model {reference!r}")
            table["skill"] = 1.0 - table["rmsle"] / table.loc[reference, "rmsle"]
        return table

    def by_step(self, metric: str = "rmsle") -> pd.DataFrame:
        """Return ``metric`` per forecast step, one column per model."""
        return self._pivot("step", metric)

    def by_origin(self, metric: str = "rmsle") -> pd.DataFrame:
        """Return ``metric`` per origin, one column per model."""
        return self._pivot("origin", metric)

    def _pivot(self, column: str, metric: str) -> pd.DataFrame:
        if metric not in SERIES_METRICS:
            raise ValueError(f"metric must be one of {SERIES_METRICS}")
        table = self.scores((column,))[metric].unstack("model")
        table.columns.name = None
        return table[self.models].astype(np.float64)


def rolling_origin_evaluation(
    wide: pd.DataFrame,
    forecasters: Mapping[str, SeriesForecaster],
    origins: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    horizon: int | None = None,
    season: int = SEASON_LENGTH,
) -> RollingResult:
    """Refit every forecaster at every origin and collect its errors.

    Parameters
    ----------
    wide : pandas.DataFrame
        Monthly sales in units, a complete monthly grid as index and one
        column per segment.
    forecasters : mapping of str to SeriesForecaster
        Unfitted forecasters; each is copied before every fit.
    origins : sequence of timestamps
        Last month of each fitting history, for example from
        :func:`rolling_origins`.
    horizon : int, optional
        Months scored after each origin. Defaults to the smallest horizon of
        the forecasters.
    season : int, default 12
        Season length of the MASE scale.

    Returns
    -------
    RollingResult
        Errors of every model, origin, segment and step.
    """
    if not forecasters:
        raise ValueError("at least one forecaster is required")
    horizon = horizon or min(model.horizon for model in forecasters.values())
    origins = pd.DatetimeIndex(origins)
    index = pd.DatetimeIndex(wide.index)
    if not origins.isin(index).all():
        raise ValueError("every origin must be a month of wide")
    if (index.get_indexer(origins) + horizon >= len(index)).any():
        raise ValueError(f"every origin needs {horizon} observed months after it")
    frames = [
        _origin_errors(name, copy.deepcopy(model), wide, origin, horizon, season)
        for origin in origins
        for name, model in forecasters.items()
    ]
    errors = pd.concat(frames, ignore_index=True)
    return RollingResult(errors=errors, origins=origins, horizon=horizon)


def paired_origin_tests(
    result: RollingResult, reference: str, metric: str = "rmsle"
) -> pd.DataFrame:
    """Compare each model with ``reference`` origin by origin.

    Returns
    -------
    pandas.DataFrame
        One row per other model with ``n_origins``, ``n_better``,
        ``median_diff`` (median over origins of the model score minus the
        reference score), the signed-rank ``statistic``, ``p_value``, the
        smallest attainable ``p_min`` and the Holm-adjusted ``p_holm``.
    """
    table = result.by_origin(metric)
    if reference not in table.columns:
        raise ValueError(f"unknown reference model {reference!r}")
    if len(table.columns) < 2:
        raise ValueError("a paired comparison needs at least two models")
    rows = {}
    for name in table.columns:
        if name == reference:
            continue
        diff = (table[name] - table[reference]).to_numpy(dtype=np.float64)
        test = signed_rank_test(diff)
        rows[name] = {
            "reference": reference,
            "n_origins": int(diff.size),
            "n_better": int(np.sum(diff < 0)),
            "median_diff": float(np.median(diff)),
            "statistic": test.statistic,
            "p_value": test.p_value,
            "p_min": test.p_min,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
    return out.sort_values("p_value")
