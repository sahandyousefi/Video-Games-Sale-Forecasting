import math
from itertools import pairwise

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vgsales.features.tabular import TabularArrays  # noqa: E402
from vgsales.models.neural.data import TabularDataset  # noqa: E402
from vgsales.models.neural.mlp import EntityEmbeddingMLP  # noqa: E402
from vgsales.models.neural.training import (  # noqa: E402
    TrainingConfig,
    evaluate_loss,
    predict,
    seed_everything,
    train_model,
)


def make_dataset(n: int, seed: int, noise_only: bool = False) -> TabularDataset:
    rng = np.random.default_rng(seed)
    categorical = rng.integers(0, 4, size=(n, 1))
    numeric = rng.normal(size=(n, 2)).astype(np.float32)
    effect = np.array([-1.0, 0.0, 0.5, 1.5])[categorical[:, 0]]
    signal = effect + 0.8 * numeric[:, 0] - 0.5 * numeric[:, 1]
    target = rng.normal(size=n) if noise_only else signal + 0.05 * rng.normal(size=n)
    return TabularDataset(TabularArrays(categorical, numeric), target)


def make_model(seed: int = 0, dropout: float = 0.0) -> EntityEmbeddingMLP:
    seed_everything(seed)
    return EntityEmbeddingMLP((4,), 2, hidden=(16,), dropout=dropout)


def test_training_fits_a_learnable_target() -> None:
    train, val = make_dataset(400, 1), make_dataset(200, 2)
    model = make_model()
    before = evaluate_loss(model, val)
    config = TrainingConfig(max_epochs=60, batch_size=32, learning_rate=1e-2)
    history = train_model(model, train, val, config)
    assert history.best_val_loss < 0.1 * before
    assert history.best_val_loss == pytest.approx(evaluate_loss(model, val))
    assert not model.training


def test_early_stopping_restores_the_best_epoch() -> None:
    train, val = make_dataset(200, 1), make_dataset(200, 2, noise_only=True)
    config = TrainingConfig(max_epochs=100, batch_size=32, patience=3)
    history = train_model(make_model(), train, val, config)
    assert history.stopped_early
    assert history.n_epochs < 100
    assert history.n_epochs - history.best_epoch == 3
    best = int(np.argmin(history.val_loss)) + 1
    assert history.best_epoch == best
    assert history.best_val_loss == min(history.val_loss)


def test_learning_rate_is_reduced_on_a_plateau() -> None:
    train, val = make_dataset(200, 1), make_dataset(200, 2, noise_only=True)
    config = TrainingConfig(
        max_epochs=40,
        batch_size=32,
        patience=40,
        lr_patience=1,
        lr_factor=0.5,
        min_lr=1e-4,
    )
    history = train_model(make_model(), train, val, config)
    rates = np.array(history.learning_rate)
    assert rates[0] == config.learning_rate
    assert (np.diff(rates) <= 0).all()
    assert rates[-1] < config.learning_rate
    assert rates.min() >= config.min_lr
    for old, new in pairwise(rates):
        assert new == old or new == pytest.approx(max(0.5 * old, config.min_lr))


def test_runs_with_the_same_seed_are_identical() -> None:
    train, val = make_dataset(300, 1), make_dataset(100, 2)
    config = TrainingConfig(max_epochs=8, batch_size=32, seed=5)
    first, second = make_model(seed=3, dropout=0.3), make_model(seed=3, dropout=0.3)
    h1 = train_model(first, train, val, config)
    h2 = train_model(second, train, val, config)
    assert h1.train_loss == h2.train_loss
    assert h1.val_loss == h2.val_loss
    for p1, p2 in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(p1, p2)

    other = make_model(seed=3, dropout=0.3)
    h3 = train_model(other, train, val, TrainingConfig(8, 32, seed=6))
    assert h3.train_loss != h1.train_loss


def test_training_without_validation_follows_the_schedule() -> None:
    train = make_dataset(200, 1)
    schedule = [1e-2, 1e-2, 5e-3, 2.5e-3]
    history = train_model(make_model(), train, lr_schedule=schedule)
    assert history.learning_rate == schedule
    assert history.n_epochs == history.best_epoch == 4
    assert history.val_loss == []
    assert math.isnan(history.best_val_loss)
    fixed = train_model(make_model(), train, config=TrainingConfig(max_epochs=3))
    assert fixed.n_epochs == 3
    assert not fixed.stopped_early


def test_predict_is_deterministic_and_ignores_the_target() -> None:
    dataset = make_dataset(50, 1)
    model = make_model(dropout=0.5)
    train_model(model, dataset, config=TrainingConfig(max_epochs=2))
    first = predict(model, dataset, batch_size=16)
    assert first.shape == (50,)
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first, predict(model, dataset))
    inputs_only = TabularDataset(
        TabularArrays(dataset.categorical.numpy(), dataset.numeric.numpy())
    )
    np.testing.assert_allclose(predict(model, inputs_only), first, rtol=1e-6)


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_epochs"):
        TrainingConfig(max_epochs=0)
    with pytest.raises(ValueError, match="lr_factor"):
        TrainingConfig(lr_factor=1.0)
    with pytest.raises(ValueError, match="grad_clip"):
        TrainingConfig(grad_clip=0.0)
    train, val = make_dataset(20, 1), make_dataset(20, 2)
    with pytest.raises(ValueError, match="only without a validation set"):
        train_model(make_model(), train, val, lr_schedule=[1e-3])
    with pytest.raises(ValueError, match="positive"):
        train_model(make_model(), train, lr_schedule=[])
