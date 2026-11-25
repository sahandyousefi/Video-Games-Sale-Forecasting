r"""Statistical baselines for the monthly aggregate series.

Both forecasters take the wide series of :func:`vgsales.data.series.to_wide`
(a complete monthly grid as index, one column per segment, sales in units)
and share the interface of :class:`vgsales.models.neural.sequence.GRUForecaster`:
``fit(wide)`` and ``forecast(history=None)``, which returns the ``horizon``
months after the end of ``history``.

Seasonal naive. With season length :math:`m` and origin :math:`T`, the
forecast repeats the last observed season,

.. math::

    \hat S_{T+h} = S_{T + h - m \lceil h / m \rceil}, \qquad h = 1, \dots, H.

Exponential smoothing. On the log scale :math:`z_t = \log(1 + S_t)`, the
additive seasonal model without trend, ETS(A,N,A) in the taxonomy of
Hyndman et al. (2008) and the seasonal method of Winters (1960) without its
trend term, keeps a level :math:`\ell_t` and seasonal states :math:`s_t`
updated in error-correction form,

.. math::

    e_t &= z_t - (\ell_{t-1} + s_{t-m}), \\
    \ell_t &= \ell_{t-1} + \alpha e_t, \\
    s_t &= s_{t-m} + \gamma e_t,

with :math:`0 < \alpha < 1` and :math:`0 \le \gamma \le 1 - \alpha`. The
forecast is :math:`\hat z_{T+h} = \ell_T + s_{T + h - m \lceil h / m \rceil}`,
mapped back by :math:`\max(0, \exp(\hat z) - 1)`. The states start from the
first season: :math:`\ell_{m-1}` is its mean and :math:`s_j = z_j -
\ell_{m-1}` for :math:`j < m`. For each segment, :math:`(\alpha, \gamma)` is
chosen on a grid by the sum of squared one-step errors over the fitting
months. With ``seasonal=False`` the seasonal states are dropped and the
method reduces to simple exponential smoothing.

The log scale turns the multiplicative seasonality of unit sales into an
additive one, and a month without tracked releases (:math:`S_t = 0`) is
the finite value :math:`z_t = 0`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

SEASON_LENGTH: Final = 12


def _as_values(wide: pd.DataFrame) -> np.ndarray:
    values = wide.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("wide must be a non-empty months by segments frame")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("series values must be finite and non-negative")
    return values


def _forecast_index(end: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(end) + pd.DateOffset(months=1)
    return pd.date_range(start, periods=horizon, freq="MS", name="period")


def _seasonal_positions(horizon: int, season: int) -> np.ndarray:
    """Offsets from the end of the series of the value repeated at each step."""
    steps = np.arange(1, horizon + 1)
    return steps - season * np.ceil(steps / season).astype(np.int64)


class _BaseForecaster:
    def __init__(self, horizon: int, season: int) -> None:
        self.horizon = horizon
        self.season = season

    def _check_params(self) -> None:
        if self.horizon < 1 or self.season < 1:
            raise ValueError("horizon and season must be >= 1")

    def _check_fitted(self) -> None:
        if not hasattr(self, "segments_"):
            raise RuntimeError("the forecaster is not fitted")

    def _history(self, history: pd.DataFrame | None) -> pd.DataFrame:
        self._check_fitted()
        if history is None:
            return self._fit_history
        missing = self.segments_.difference(history.columns)
        if len(missing):
            raise ValueError(f"history is missing fitted segments: {list(missing)}")
        return history[self.segments_]


class SeasonalNaiveForecaster(_BaseForecaster):
    """Repeat the last observed season.

    Parameters
    ----------
    horizon : int, default 12
        Forecast months :math:`H`.
    season : int, default 12
        Season length :math:`m` in months.
    """

    def __init__(self, horizon: int = 12, season: int = SEASON_LENGTH) -> None:
        super().__init__(horizon, season)

    def fit(self, wide: pd.DataFrame) -> SeasonalNaiveForecaster:
        """Store the segments and the end of ``wide``."""
        self._check_params()
        if len(wide) < self.season:
            raise ValueError(f"wide needs at least {self.season} months")
        _as_values(wide)
        self.segments_ = pd.Index(wide.columns)
        self._fit_history = wide
        return self

    def forecast(self, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Forecast the ``horizon`` months after the end of ``history``."""
        history = self._history(history)
        values = _as_values(history)
        if len(values) < self.season:
            raise ValueError(f"history needs at least {self.season} months")
        offsets = _seasonal_positions(self.horizon, self.season)
        forecast = values[len(values) - 1 + offsets]
        return pd.DataFrame(
            forecast,
            index=_forecast_index(history.index[-1], self.horizon),
            columns=self.segments_,
        )


def _smoothing_grid(
    n_alpha: int, n_gamma: int, seasonal: bool
) -> tuple[np.ndarray, np.ndarray]:
    alphas = np.linspace(0.02, 0.98, n_alpha)
    if not seasonal:
        return alphas, np.zeros_like(alphas)
    grid_alpha, grid_gamma = [], []
    for alpha in alphas:
        for share in np.linspace(0.0, 1.0, n_gamma):
            grid_alpha.append(alpha)
            grid_gamma.append(share * (1.0 - alpha))
    return np.asarray(grid_alpha), np.asarray(grid_gamma)


def _run_filter(
    z: np.ndarray, alpha: np.ndarray, gamma: np.ndarray, season: int, seasonal: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the recursions for every (alpha, gamma) pair and segment.

    Parameters
    ----------
    z : numpy.ndarray, shape (n_months, n_segments)
        Log-scale series.
    alpha, gamma : numpy.ndarray, shape (n_grid,) or (n_grid, n_segments)
        Smoothing parameters.

    Returns
    -------
    sse : numpy.ndarray, shape (n_grid, n_segments)
        Sum of squared one-step errors after the first season.
    level : numpy.ndarray, shape (n_grid, n_segments)
        Final level :math:`\\ell_T`.
    states : numpy.ndarray, shape (n_grid, season, n_segments)
        Last ``season`` seasonal states, oldest first.
    """
    n_months, n_segments = z.shape
    alpha = alpha[:, None] if alpha.ndim == 1 else alpha
    gamma = gamma[:, None] if gamma.ndim == 1 else gamma
    n_grid = alpha.shape[0]
    if seasonal:
        first = z[:season].mean(axis=0)
        level = np.tile(first, (n_grid, 1))
        states = np.tile(z[:season] - first, (n_grid, 1, 1))
        start = season
    else:
        level = np.tile(z[0], (n_grid, 1))
        states = np.zeros((n_grid, season, n_segments))
        start = 1
    sse = np.zeros((n_grid, n_segments))
    for t in range(start, n_months):
        # Circular buffer: s_{t-m} sits at position t mod m and is replaced by s_t.
        slot = t % season
        error = z[t] - (level + states[:, slot])
        sse += error**2
        level = level + alpha * error
        if seasonal:
            states[:, slot] += gamma * error
    states = np.roll(states, -(n_months % season), axis=1)
    return sse, level, states


class ExponentialSmoothingForecaster(_BaseForecaster):
    """Additive seasonal exponential smoothing on ``log1p`` sales.

    Parameters
    ----------
    horizon : int, default 12
        Forecast months :math:`H`.
    season : int, default 12
        Season length :math:`m` in months.
    seasonal : bool, default True
        Keep the seasonal states; False gives simple exponential smoothing.
    n_alpha, n_gamma : int, default 25 and 11
        Grid sizes for :math:`\\alpha` and for the share of
        :math:`1 - \\alpha` taken by :math:`\\gamma`.

    Attributes
    ----------
    alpha_, gamma_ : numpy.ndarray
        Selected smoothing parameters of each segment.
    level_ : numpy.ndarray
        Level :math:`\\ell_T` of each segment at the end of the fitting data.
    seasonal_ : numpy.ndarray, shape (season, n_segments)
        Last ``season`` seasonal states, oldest first.
    """

    def __init__(
        self,
        horizon: int = 12,
        season: int = SEASON_LENGTH,
        seasonal: bool = True,
        n_alpha: int = 25,
        n_gamma: int = 11,
    ) -> None:
        super().__init__(horizon, season)
        self.seasonal = seasonal
        self.n_alpha = n_alpha
        self.n_gamma = n_gamma

    def fit(self, wide: pd.DataFrame) -> ExponentialSmoothingForecaster:
        """Select the smoothing parameters of each segment on ``wide``."""
        self._check_params()
        if self.n_alpha < 1 or self.n_gamma < 1:
            raise ValueError("n_alpha and n_gamma must be >= 1")
        z = np.log1p(_as_values(wide))
        min_months = 2 * self.season if self.seasonal else 2
        if len(z) < min_months:
            raise ValueError(f"wide needs at least {min_months} months")
        alphas, gammas = _smoothing_grid(self.n_alpha, self.n_gamma, self.seasonal)
        sse, level, states = _run_filter(z, alphas, gammas, self.season, self.seasonal)
        best = np.argmin(sse, axis=0)
        segments = np.arange(z.shape[1])
        self.alpha_ = alphas[best]
        self.gamma_ = gammas[best]
        self.level_ = level[best, segments]
        self.seasonal_ = states[best, :, segments].T
        self.segments_ = pd.Index(wide.columns)
        self._fit_history = wide
        return self

    def _states(self, history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if history is self._fit_history:
            return self.level_, self.seasonal_
        z = np.log1p(_as_values(history))
        min_months = 2 * self.season if self.seasonal else 2
        if len(z) < min_months:
            raise ValueError(f"history needs at least {min_months} months")
        _, level, states = _run_filter(
            z, self.alpha_[None, :], self.gamma_[None, :], self.season, self.seasonal
        )
        return level[0], states[0]

    def forecast(self, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Forecast the ``horizon`` months after the end of ``history``.

        A ``history`` other than the fitting series is filtered again with
        the fitted smoothing parameters, so forecasts can be issued from a
        later origin without reselecting them.
        """
        history = self._history(history)
        level, states = self._states(history)
        offsets = _seasonal_positions(self.horizon, self.season)
        # states[-1] is the state of the last month, offset 0.
        z = level[None, :] + states[self.season - 1 + offsets]
        return pd.DataFrame(
            np.maximum(np.expm1(z), 0.0),
            index=_forecast_index(history.index[-1], self.horizon),
            columns=self.segments_,
        )
