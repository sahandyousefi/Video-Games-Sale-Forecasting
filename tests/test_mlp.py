import pytest

torch = pytest.importorskip("torch")

from vgsales.models.neural.mlp import (  # noqa: E402
    EntityEmbeddingMLP,
    embedding_dim,
    parameter_groups,
)


def batch(n: int = 8, cardinalities: tuple[int, ...] = (5, 30), d: int = 3):
    generator = torch.Generator().manual_seed(0)
    categorical = torch.stack(
        [torch.randint(0, k, (n,), generator=generator) for k in cardinalities], dim=1
    )
    return categorical, torch.randn(n, d, generator=generator)


def test_embedding_width_heuristic() -> None:
    assert embedding_dim(1) == 1
    assert embedding_dim(5) == 3
    assert embedding_dim(30) == 15
    assert embedding_dim(500) == 16
    assert embedding_dim(500, max_dim=40) == 40
    with pytest.raises(ValueError):
        embedding_dim(0)


def test_forward_shape_and_layout() -> None:
    torch.manual_seed(0)
    model = EntityEmbeddingMLP((5, 30), n_numeric=3, hidden=(16, 8))
    assert model.embedding_dims == (3, 15)
    assert model.body[0].in_features == 3 + 15 + 3
    out = model(*batch())
    assert out.shape == (8,)
    assert out.dtype == torch.float32


def test_numeric_only_and_categorical_only_inputs() -> None:
    categorical, numeric = batch()
    numeric_only = EntityEmbeddingMLP((), n_numeric=3, hidden=(4,))
    assert numeric_only(categorical[:, :0], numeric).shape == (8,)
    categorical_only = EntityEmbeddingMLP((5, 30), n_numeric=0, hidden=(4,))
    assert categorical_only(categorical, numeric[:, :0]).shape == (8,)
    with pytest.raises(ValueError, match="no inputs"):
        EntityEmbeddingMLP((), n_numeric=0)


def test_dropout_is_active_in_training_mode_only() -> None:
    torch.manual_seed(0)
    model = EntityEmbeddingMLP((5, 30), 3, hidden=(64,), dropout=0.5)
    inputs = batch()
    model.train()
    assert not torch.equal(model(*inputs), model(*inputs))
    model.eval()
    assert torch.equal(model(*inputs), model(*inputs))


def test_gradients_reach_only_the_embedding_rows_in_the_batch() -> None:
    torch.manual_seed(0)
    model = EntityEmbeddingMLP((5, 30), 3, hidden=(8,), dropout=0.0)
    categorical, numeric = batch()
    model(categorical, numeric).sum().backward()
    grad = model.embeddings[1].weight.grad
    used = set(categorical[:, 1].tolist())
    for index in range(30):
        assert bool(grad[index].abs().sum() > 0) == (index in used)


def test_output_bias_initialization() -> None:
    model = EntityEmbeddingMLP((5,), 3, hidden=(4,))
    model.init_output_bias(-0.7)
    with torch.no_grad():
        model.head.weight.zero_()
    categorical, numeric = batch(cardinalities=(5,))
    torch.testing.assert_close(model(categorical, numeric), torch.full((8,), -0.7))


def test_parameter_groups_exclude_biases_from_decay() -> None:
    model = EntityEmbeddingMLP((5, 30), 3, hidden=(16, 8))
    decay, no_decay = parameter_groups(model, weight_decay=1e-4)
    assert decay["weight_decay"] == 1e-4
    assert no_decay["weight_decay"] == 0.0
    assert all(p.ndim == 2 for p in decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"])
    grouped = {id(p) for p in decay["params"] + no_decay["params"]}
    assert grouped == {id(p) for p in model.parameters()}
    assert len(decay["params"]) + len(no_decay["params"]) == len(grouped)
    optimizer = torch.optim.AdamW([decay, no_decay], lr=1e-3)
    assert len(optimizer.param_groups) == 2


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="dropout"):
        EntityEmbeddingMLP((5,), 3, dropout=1.0)
    with pytest.raises(ValueError, match="embedding_dims"):
        EntityEmbeddingMLP((5, 6), 3, embedding_dims=(2,))
    with pytest.raises(ValueError, match="weight_decay"):
        parameter_groups(EntityEmbeddingMLP((5,), 3), weight_decay=-1.0)
