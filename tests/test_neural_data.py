import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vgsales.features.tabular import TabularArrays  # noqa: E402
from vgsales.models.neural.data import TabularDataset, make_loader  # noqa: E402


def make_arrays(n: int = 10) -> TabularArrays:
    categorical = np.stack([np.arange(n) % 3, np.arange(n) % 4], axis=1)
    numeric = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    return TabularArrays(categorical.astype(np.int64), numeric)


def test_dataset_returns_rows_and_batches() -> None:
    target = np.linspace(0.0, 1.0, 10)
    dataset = TabularDataset(make_arrays(), target)
    assert len(dataset) == 10
    assert dataset.has_target
    categorical, numeric, y = dataset[3]
    assert categorical.dtype == torch.long
    assert numeric.dtype == torch.float32
    assert y.dtype == torch.float32
    assert categorical.tolist() == [0, 3]
    categorical, numeric, y = dataset[[1, 4, 7]]
    assert categorical.shape == (3, 2)
    assert numeric.shape == (3, 2)
    np.testing.assert_allclose(y.numpy(), target[[1, 4, 7]], rtol=1e-6)


def test_dataset_without_target_yields_inputs_only() -> None:
    dataset = TabularDataset(make_arrays())
    assert not dataset.has_target
    assert len(dataset[[0, 1]]) == 2


def test_dataset_rejects_a_bad_target() -> None:
    with pytest.raises(ValueError, match="one value per row"):
        TabularDataset(make_arrays(), np.zeros(9))
    with pytest.raises(ValueError, match="finite"):
        TabularDataset(make_arrays(), np.full(10, np.nan))


def test_sequential_loader_keeps_the_row_order() -> None:
    dataset = TabularDataset(make_arrays(), np.arange(10.0))
    batches = list(make_loader(dataset, batch_size=4))
    assert [len(batch[2]) for batch in batches] == [4, 4, 2]
    order = torch.cat([batch[2] for batch in batches]).tolist()
    assert order == list(range(10))
    assert len(list(make_loader(dataset, batch_size=4, drop_last=True))) == 2


def epoch_orders(seed: int, epochs: int = 3) -> list[list[float]]:
    dataset = TabularDataset(make_arrays(), np.arange(10.0))
    loader = make_loader(dataset, batch_size=3, shuffle=True, seed=seed)
    return [torch.cat([b[2] for b in loader]).tolist() for _ in range(epochs)]


def test_shuffled_loader_is_reproducible_per_seed() -> None:
    first = epoch_orders(seed=11)
    assert first == epoch_orders(seed=11)
    assert first != epoch_orders(seed=12)
    for order in first:
        assert sorted(order) == list(range(10))
    assert first[0] != first[1]


def test_shuffled_loader_ignores_the_global_random_state() -> None:
    torch.manual_seed(0)
    first = epoch_orders(seed=3, epochs=1)
    torch.manual_seed(1)
    assert first == epoch_orders(seed=3, epochs=1)


def test_loader_arguments_are_checked() -> None:
    dataset = TabularDataset(make_arrays())
    with pytest.raises(ValueError, match="seed"):
        make_loader(dataset, shuffle=True)
    with pytest.raises(ValueError, match="batch_size"):
        make_loader(dataset, batch_size=0)
