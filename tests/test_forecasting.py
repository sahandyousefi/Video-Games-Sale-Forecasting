import numpy as np
import pandas as pd
import pytest

from vgsales.models.forecasting import (
    ExponentialSmoothingForecaster,
    SeasonalNaiveForecaster,
)


def make_wide(values: np.ndarray, start: str = "2000-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(values), freq="MS")
    columns = [f"seg{k}" for k in range(values.shape[1])]
    return pd.DataFrame(values, index=index, columns=columns)


def seasonal_values(n_months: int, n_segments: int, noise: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    month = np.arange(n_months)[:, None] % 12
    phase = rng.uniform(0, 2 * np.pi, size=n_segments)
    z = 1.5 + 0.7 * np.sin(2 * np.pi * month / 12 + phase)
    z = z + noise * rng.normal(size=(n_months, n_segments))
    return np.expm1(z)


def reference_ets(z: np.ndarray, alpha: float, gamma: float, season: int, h: int):
    level = z[:season].mean()
    states = list(z[:season] - level)
    for t in range(season, len(z)):
        error = z[t] - (level + states[t - season])
        level += alpha * error
        states.append(states[t - season] + gamma * error)
    n = len(z)
    return np.array(
        [level + states[n - 1 + k - season * int(np.ceil(k / season))] for k in h]
    )


def test_seasonal_naive_repeats_the_last_season() -> None:
    values = np.arange(36, dtype=float).reshape(18, 2)
    wide = make_wide(values)
    model = SeasonalNaiveForecaster(horizon=18, season=6).fit(wide)
    forecast = model.forecast()
    last_season = values[-6:]
    np.testing.assert_array_equal(forecast.to_numpy(), np.tile(last_season, (3, 1)))
    assert forecast.index[0] == wide.index[-1] + pd.DateOffset(months=1)
    assert list(forecast.columns) == list(wide.columns)

    later = model.forecast(make_wide(values[:12] + 100.0))
    np.testing.assert_array_equal(later.to_numpy()[:6], values[6:12] + 100.0)


def test_ets_matches_the_reference_recursion() -> None:
    wide = make_wide(seasonal_values(60, 3, noise=0.2))
    model = ExponentialSmoothingForecaster(horizon=15).fit(wide)
    z = np.log1p(wide.to_numpy())
    forecast = model.forecast()
    for k in range(3):
        expected = reference_ets(
            z[:, k], model.alpha_[k], model.gamma_[k], 12, range(1, 16)
        )
        np.testing.assert_allclose(
            np.log1p(forecast.iloc[:, k].to_numpy()), expected, atol=1e-10
        )
    assert ((model.gamma_ >= 0) & (model.gamma_ <= 1 - model.alpha_ + 1e-12)).all()


def test_ets_recovers_a_stable_seasonal_pattern() -> None:
    values = seasonal_values(96, 2, noise=0.0)
    wide = make_wide(values)
    forecast = ExponentialSmoothingForecaster(horizon=12).fit(wide.iloc[:-12])
    np.testing.assert_allclose(forecast.forecast().to_numpy(), values[-12:], rtol=1e-8)


def test_ets_refilters_a_later_history_with_fitted_parameters() -> None:
    values = seasonal_values(72, 2, noise=0.3, seed=4)
    wide = make_wide(values)
    model = ExponentialSmoothingForecaster(horizon=6).fit(wide.iloc[:48])
    later = model.forecast(wide)
    assert later.index[0] == wide.index[-1] + pd.DateOffset(months=1)
    z = np.log1p(values)
    for k in range(2):
        expected = reference_ets(
            z[:, k], model.alpha_[k], model.gamma_[k], 12, range(1, 7)
        )
        np.testing.assert_allclose(np.log1p(later.iloc[:, k]), expected, atol=1e-10)


def test_simple_smoothing_tracks_a_random_walk() -> None:
    rng = np.random.default_rng(2)
    z = 2.0 + np.cumsum(0.3 * rng.normal(size=(120, 1)), axis=0)
    wide = make_wide(np.expm1(np.abs(z)))
    model = ExponentialSmoothingForecaster(horizon=3, seasonal=False).fit(wide)
    assert model.alpha_[0] > 0.7
    assert (model.gamma_ == 0).all()
    forecast = model.forecast().to_numpy()[:, 0]
    assert np.allclose(forecast, forecast[0])


def test_invalid_inputs_are_rejected() -> None:
    wide = make_wide(seasonal_values(30, 2, noise=0.1))
    with pytest.raises(RuntimeError, match="not fitted"):
        SeasonalNaiveForecaster().forecast(wide)
    with pytest.raises(ValueError, match="at least 24 months"):
        ExponentialSmoothingForecaster().fit(wide.iloc[:20])
    with pytest.raises(ValueError, match="non-negative"):
        SeasonalNaiveForecaster().fit(wide - 100.0)
    with pytest.raises(ValueError, match=">= 1"):
        SeasonalNaiveForecaster(horizon=0).fit(wide)
    model = ExponentialSmoothingForecaster().fit(wide)
    with pytest.raises(ValueError, match="missing fitted segments"):
        model.forecast(wide.drop(columns="seg0"))
