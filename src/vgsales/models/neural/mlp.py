r"""Multilayer perceptron with entity embeddings of the categorical inputs.

Following Guo and Berkhahn (2016), each categorical variable :math:`j` with
:math:`K_j` levels is mapped to a learned vector by an embedding table
:math:`E_j \in \mathbb{R}^{K_j \times q_j}`. For a row with embedding indices
:math:`c_1, \dots, c_m` and standardized numeric inputs
:math:`u \in \mathbb{R}^d`, the network computes

.. math::

    h_0 &= \big[\, E_1[c_1], \dots, E_m[c_m], u \,\big]
          \in \mathbb{R}^{q_1 + \dots + q_m + d}, \\
    h_l &= \operatorname{dropout}_p\big(\operatorname{ReLU}(W_l h_{l-1}
          + b_l)\big), \quad l = 1, \dots, L, \\
    \hat z &= w^\top h_L + b,

where :math:`\hat z` is the prediction on the modelling scale. Dropout
(Srivastava et al., 2014) with rate :math:`p` is applied after every hidden
layer and, with its own rate, to the concatenated embeddings. Compared with
one-hot inputs, the embeddings let publishers and developers with similar
sales histories share statistical strength through nearby vectors, and the
number of first-layer weights grows with :math:`q_j` rather than with
:math:`K_j`.

Weight decay is applied as decoupled weight decay (Loshchilov and Hutter,
2019) to the weight matrices and embedding tables. Biases are excluded,
since shrinking them toward zero only shifts the predictions.
:func:`parameter_groups` builds the corresponding optimizer groups.

The embedding width follows the heuristic
:math:`q_j = \min(q_{\max}, \lceil K_j / 2 \rceil)`, which gives small
tables to variables with few levels and caps the width of high-cardinality
variables.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def embedding_dim(cardinality: int, max_dim: int = 16) -> int:
    """Return the embedding width for a variable with ``cardinality`` levels.

    Parameters
    ----------
    cardinality : int
        Size of the embedding table, out-of-vocabulary index included.
    max_dim : int, default 16
        Upper bound :math:`q_{\\max}` on the width.

    Returns
    -------
    int
        :math:`\\min(q_{\\max}, \\lceil K / 2 \\rceil)`, at least 1.
    """
    if cardinality < 1 or max_dim < 1:
        raise ValueError("cardinality and max_dim must be >= 1")
    return max(1, min(max_dim, math.ceil(cardinality / 2)))


class EntityEmbeddingMLP(nn.Module):
    """Feed-forward regressor on entity embeddings and numeric inputs.

    Parameters
    ----------
    cardinalities : sequence of int
        Embedding table size of each categorical column, in input order.
    n_numeric : int
        Number of numeric inputs.
    hidden : sequence of int, default (128, 64)
        Width of each hidden layer.
    dropout : float, default 0.1
        Dropout rate after each hidden layer.
    embedding_dropout : float, default 0.0
        Dropout rate on the concatenated embeddings.
    max_embedding_dim : int, default 16
        Cap on the embedding width; see :func:`embedding_dim`.
    embedding_dims : sequence of int, optional
        Explicit embedding widths, overriding the heuristic.
    """

    def __init__(
        self,
        cardinalities: Sequence[int],
        n_numeric: int,
        hidden: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
        max_embedding_dim: int = 16,
        embedding_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        cardinalities = [int(k) for k in cardinalities]
        if embedding_dims is None:
            dims = [embedding_dim(k, max_embedding_dim) for k in cardinalities]
        else:
            dims = [int(q) for q in embedding_dims]
            if len(dims) != len(cardinalities):
                raise ValueError("embedding_dims must match cardinalities")
        if n_numeric < 0:
            raise ValueError("n_numeric must be >= 0")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= embedding_dropout < 1.0:
            raise ValueError("dropout rates must lie in [0, 1)")
        if any(width < 1 for width in hidden):
            raise ValueError("hidden layer widths must be >= 1")
        n_inputs = sum(dims) + n_numeric
        if n_inputs == 0:
            raise ValueError("the network has no inputs")

        self.cardinalities = tuple(cardinalities)
        self.n_numeric = int(n_numeric)
        self.embeddings = nn.ModuleList(
            nn.Embedding(k, q) for k, q in zip(cardinalities, dims, strict=True)
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        layers: list[nn.Module] = []
        width = n_inputs
        for out in hidden:
            layers += [nn.Linear(width, out), nn.ReLU(), nn.Dropout(dropout)]
            width = out
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    @property
    def embedding_dims(self) -> tuple[int, ...]:
        """Width of each embedding table."""
        return tuple(e.embedding_dim for e in self.embeddings)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        """Predict the modelling-scale target of a batch.

        Parameters
        ----------
        categorical : torch.Tensor of int64, shape (n, m)
            Embedding indices.
        numeric : torch.Tensor of float32, shape (n, d)
            Standardized numeric inputs.

        Returns
        -------
        torch.Tensor of float32, shape (n,)
        """
        parts = [
            embedding(categorical[:, j]) for j, embedding in enumerate(self.embeddings)
        ]
        if parts:
            embedded = self.embedding_dropout(torch.cat(parts, dim=1))
            inputs = torch.cat([embedded, numeric], dim=1)
        else:
            inputs = numeric
        return self.head(self.body(inputs)).squeeze(-1)

    @torch.no_grad()
    def init_output_bias(self, value: float) -> None:
        """Set the output bias, typically to the mean training target.

        Starting from the mean removes the initial offset that the first
        epochs would otherwise spend learning.
        """
        self.head.bias.fill_(float(value))


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split the parameters of ``model`` into decayed and undecayed groups.

    Parameters
    ----------
    model : torch.nn.Module
        Network whose parameters are optimized.
    weight_decay : float
        Decay coefficient of the weight matrices and embedding tables.

    Returns
    -------
    list of dict
        Two optimizer parameter groups: the weights of linear and embedding
        layers with ``weight_decay``, and all other parameters (biases) with
        zero decay. Each trainable parameter appears in exactly one group.
    """
    if weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    decay, no_decay = [], []
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            weighted = isinstance(module, nn.Linear | nn.Embedding) and name == "weight"
            (decay if weighted else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
