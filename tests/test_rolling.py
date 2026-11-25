from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.series import aggregate_series, to_wide, uncensored_end
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.evaluation.rolling import (
    ERROR_COLUMNS,
    SERIES_METRICS,
    paired_origin_tests,
    rolling_origin_evaluation,
    rolling_origins,
    seasonal_scale,
)
from vgsales.models.forecasting import (
    FORECASTER_NAMES,
    ExponentialSmoothingForecaster,
    SeasonalNaiveForecaster,
    make_forecaster,
)


def periodic_wide(n_months: int = 96, n_segments: int = 3, noise: float = 0.0):
    rng = np.random.default_rng(0)
    index = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    month = index.month.to_numpy()[:, None]
    phase = rng.uniform(0, 2 * np.pi, size=n_segments)
    z = 1.5 + 0.6 * np.sin(2 * np.pi * month / 12 + phase)
    z = z + noise * rng.normal(size=z.shape)
    columns = [f"seg{k}" for k in range(n_segments)]
    return pd.DataFrame(np.expm1(z), index=index, columns=columns)


class SpyForecaster(SeasonalNaiveForecaster):
    seen: ClassVar[list[pd.Timestamp]] = []

    def fit(self, wide: pd.DataFrame) -> "SpyForecaster":
        SpyForecaster.seen.append(wide.index[-1])
        return super().fit(wide)


def test_rolling_origins() -> None:
    index = pd.date_range("2000-01-01", periods=60, freq="MS")
    origins = rolling_origins(index, horizon=6, n_origins=3, step=12, min_history=30)
    assert list(origins) == list(index[[29, 41, 53]])
    with pytest.raises(ValueError, match="too few"):
        rolling_origins(index, horizon=6, n_origins=3, step=12, min_history=31)


def test_no_forecaster_sees_months_after_its_origin() -> None:
    wide = periodic_wide(noise=0.1)
    origins = rolling_origins(wide.index, 6, 3, step=6)
    SpyForecaster.seen = []
    result = rolling_origin_evaluation(wide, {"spy": SpyForecaster(6)}, origins)
    assert SpyForecaster.seen == list(origins)
    assert list(result.errors.columns) == list(ERROR_COLUMNS)
    assert len(result.errors) == 3 * 3 * 6
    first = result.errors[result.errors["origin"] == origins[0]]
    expected = wide.loc[origins[0] + pd.DateOffset(months=1) :].iloc[:6]
    np.testing.assert_allclose(
        first.sort_values(["step", "segment"])["actual"], expected.to_numpy().ravel()
    )


def test_seasonal_naive_is_exact_on_a_periodic_series() -> None:
    wide = periodic_wide()
    origins = rolling_origins(wide.index, 12, 4, step=6)
    models = {
        "seasonal_naive": SeasonalNaiveForecaster(12),
        "ses": ExponentialSmoothingForecaster(12, seasonal=False),
    }
    result = rolling_origin_evaluation(wide, models, origins)
    summary = result.summary(reference="ses")
    assert list(summary.columns) == [*SERIES_METRICS, "n", "skill"]
    assert summary.loc["seasonal_naive", "rmsle"] == pytest.approx(0.0, abs=1e-12)
    assert summary.loc["seasonal_naive", "skill"] == pytest.approx(1.0)
    assert summary.loc["ses", "skill"] == 0.0
    # The scale of a periodic series is zero, so MASE is undefined.
    assert np.isnan(summary.loc["ses", "mase"])
    assert result.by_step().shape == (12, 2)
    assert result.by_origin("mae").shape == (4, 2)
    assert result.models == ["seasonal_naive", "ses"]


def test_mase_scale_and_scores() -> None:
    wide = periodic_wide(noise=0.2)
    history = wide.iloc[:40]
    z = np.log1p(history.to_numpy())
    expected = np.abs(z[12:] - z[:-12]).mean(axis=0)
    np.testing.assert_allclose(seasonal_scale(history), expected)

    origins = rolling_origins(wide.index, 6, 2, step=12)
    result = rolling_origin_evaluation(
        wide, {"ets": ExponentialSmoothingForecaster(6)}, origins
    )
    errors = result.errors
    log_error = np.log1p(errors["forecast"]) - np.log1p(errors["actual"])
    summary = result.summary()
    assert summary.loc["ets", "rmsle"] == pytest.approx(np.sqrt(np.mean(log_error**2)))
    assert summary.loc["ets", "mase"] == pytest.approx(
        np.mean(np.abs(log_error) / errors["scale"])
    )
    per_step = result.scores(("step",))
    assert per_step["n"].sum() == len(errors)


def test_paired_origin_tests() -> None:
    wide = periodic_wide(noise=0.3)
    origins = rolling_origins(wide.index, 6, 5, step=6)
    models = {name: make_forecaster(name, 6) for name in ("seasonal_naive", "ets")}
    models["ses"] = make_forecaster("ses", 6)
    result = rolling_origin_evaluation(wide, models, origins)
    table = paired_origin_tests(result, reference="ets")
    assert set(table.index) == {"seasonal_naive", "ses"}
    assert (table["n_origins"] == 5).all()
    assert ((table["p_holm"] >= table["p_value"]) & (table["p_holm"] <= 1)).all()
    with pytest.raises(ValueError, match="unknown reference"):
        paired_origin_tests(result, reference="gru")


def test_invalid_evaluations_are_rejected() -> None:
    wide = periodic_wide()
    with pytest.raises(ValueError, match="observed months"):
        rolling_origin_evaluation(
            wide, {"naive": SeasonalNaiveForecaster(12)}, wide.index[[-3]]
        )
    with pytest.raises(ValueError, match="fewer than horizon"):
        rolling_origin_evaluation(
            wide, {"naive": SeasonalNaiveForecaster(3)}, wide.index[[40]], horizon=6
        )
    with pytest.raises(ValueError, match="unknown forecaster"):
        make_forecaster("arima", 6)
    assert "gru" in FORECASTER_NAMES


def test_gru_against_baselines_on_synthetic_series() -> None:
    pytest.importorskip("torch")
    raw = make_synthetic(SyntheticSpec(n_titles=1500), seed=5)
    config = load_config()
    titles = apply_release_scope(clean(raw)[0], config.data)[0]
    end = uncensored_end(
        titles["release_date"].max(), config.data.censoring_window_years
    )
    wide = to_wide(aggregate_series(titles, by=("genre",), end=end))
    wide = wide.loc["1995-01-01":]
    origins = rolling_origins(wide.index, 6, 2, step=12, min_history=60)
    gru = make_forecaster(
        "gru", 6, input_length=12, hidden_size=8, max_epochs=5, validation_months=6
    )
    models = {"seasonal_naive": SeasonalNaiveForecaster(6), "gru": gru}
    result = rolling_origin_evaluation(wide, models, origins)
    summary = result.summary(reference="seasonal_naive")
    assert np.isfinite(summary[["rmsle", "mae"]].to_numpy()).all()
    assert (result.errors["forecast"] >= 0).all()
