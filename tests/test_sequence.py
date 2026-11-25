import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from vgsales.features.windows import SeriesScaler, make_windows  # noqa: E402
from vgsales.models.neural.mlp import parameter_groups  # noqa: E402
from vgsales.models.neural.sequence import (  # noqa: E402
    GRUForecaster,
    GRUNetwork,
    SequenceDataset,
)
from vgsales.models.neural.training import evaluate_loss  # noqa: E402


def seasonal_wide(n_months: int = 120, n_segments: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    month = index.month.to_numpy()[:, None]
    level = rng.uniform(1.0, 2.5, size=n_segments)
    phase = rng.uniform(0, 2 * np.pi, size=n_segments)
    log_sales = level + 0.8 * np.sin(2 * np.pi * month / 12 + phase)
    log_sales = log_sales + 0.05 * rng.normal(size=(n_months, n_segments))
    columns = [f"seg{k}" for k in range(n_segments)]
    return pd.DataFrame(np.expm1(log_sales), index=index, columns=columns)


def small_forecaster(**overrides) -> GRUForecaster:
    params = dict(
        input_length=12,
        horizon=6,
        hidden_size=16,
        segment_dim=2,
        learning_rate=1e-2,
        batch_size=32,
        max_epochs=60,
        patience=10,
        validation_months=12,
    )
    params.update(overrides)
    return GRUForecaster(**params)


def test_network_shapes() -> None:
    network = GRUNetwork(n_segments=3, horizon=5, hidden_size=8, num_layers=2)
    inputs = torch.randn(7, 10, 3)
    segment = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    assert network(inputs, segment).shape == (7, 5)
    plain = GRUNetwork(n_segments=3, horizon=5, hidden_size=8, segment_dim=0)
    assert plain.segment_embedding is None
    assert plain(inputs, segment).shape == (7, 5)
    with pytest.raises(ValueError, match="dropout"):
        GRUNetwork(3, 5, dropout=1.0)


def test_recurrent_weights_are_decayed() -> None:
    network = GRUNetwork(n_segments=3, horizon=2, hidden_size=4)
    decay, no_decay = parameter_groups(network, 0.1)
    decayed = {id(p) for p in decay["params"]}
    assert id(network.gru.weight_hh_l0) in decayed
    assert id(network.gru.weight_ih_l0) in decayed
    assert id(network.gru.bias_hh_l0) not in decayed
    assert len(decay["params"]) + len(no_decay["params"]) == len(
        list(network.parameters())
    )


def test_dataset_and_multi_horizon_loss() -> None:
    wide = seasonal_wide(n_months=40, n_segments=2)
    windows = make_windows(SeriesScaler().fit(wide).transform(wide), 6, 3)
    dataset = SequenceDataset(windows)
    inputs, segment, target = dataset[[0, 1, 2]]
    assert inputs.shape == (3, 6, 3)
    assert segment.dtype == torch.long
    assert target.shape == (3, 3)

    network = GRUNetwork(n_segments=2, horizon=3, hidden_size=4)
    network.eval()
    with torch.no_grad():
        error = network(dataset.inputs, dataset.segment) - dataset.target
    expected = float((error**2).mean())
    assert evaluate_loss(network, dataset, batch_size=7) == pytest.approx(
        expected, rel=1e-5
    )


def test_forecaster_learns_the_seasonal_pattern() -> None:
    wide = seasonal_wide()
    train, test = wide.iloc[:-6], wide.iloc[-6:]
    model = small_forecaster().fit(train)
    forecast = model.forecast()
    assert list(forecast.columns) == list(wide.columns)
    assert (forecast.index == test.index).all()
    assert (forecast.to_numpy() >= 0).all()

    error_model = np.log1p(forecast.to_numpy()) - np.log1p(test.to_numpy())
    error_mean = np.log1p(train).mean().to_numpy() - np.log1p(test.to_numpy())
    rmse_model = np.sqrt(np.mean(error_model**2))
    rmse_mean = np.sqrt(np.mean(error_mean**2))
    assert rmse_model < 0.6 * rmse_mean
    assert np.isfinite(model.validation_rmse_)


def test_refit_uses_the_selected_epochs() -> None:
    wide = seasonal_wide(n_months=72)
    model = small_forecaster(max_epochs=15).fit(wide)
    assert model.selection_history_ is not None
    assert model.best_epoch_ == model.selection_history_.best_epoch
    assert model.history_.n_epochs == model.best_epoch_
    assert (
        model.history_.learning_rate
        == (model.selection_history_.learning_rate[: model.best_epoch_])
    )

    kept = small_forecaster(max_epochs=15, refit=False).fit(wide)
    assert kept.history_ is kept.selection_history_

    no_val = small_forecaster(max_epochs=4, validation_months=0).fit(wide)
    assert no_val.selection_history_ is None
    assert no_val.history_.n_epochs == 4
    assert np.isnan(no_val.validation_rmse_)
    with pytest.raises(ValueError, match=">= horizon"):
        small_forecaster(validation_months=3).fit(wide)


def test_fits_are_deterministic_and_forecast_from_later_origins() -> None:
    wide = seasonal_wide(n_months=72)
    first = small_forecaster(max_epochs=8).fit(wide.iloc[:-12])
    second = small_forecaster(max_epochs=8).fit(wide.iloc[:-12])
    pd.testing.assert_frame_equal(first.forecast(), second.forecast())

    later = first.forecast(wide.iloc[:-6])
    assert later.index[0] == wide.index[-6]
    assert len(later) == first.horizon
    with pytest.raises(RuntimeError, match="not fitted"):
        GRUForecaster().forecast(wide)
