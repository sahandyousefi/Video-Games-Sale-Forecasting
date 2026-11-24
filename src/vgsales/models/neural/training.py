r"""Mini-batch training with early stopping and learning-rate scheduling.

:func:`train_model` minimizes the mean squared error

.. math::

    \mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n}
        \big(\hat z_\theta(x_i) - z_i\big)^2

on the modelling scale of the target with AdamW (Loshchilov and Hutter,
2019), the decoupled form of weight decay of Adam (Kingma and Ba, 2015).
Under the default ``log1p`` transform, :math:`\sqrt{\mathcal{L}}` on a
validation set is its RMSLE. The datasets yield tuples whose last tensor is
the target and whose other tensors are the inputs of the model, so the same
loop trains the tabular network and the sequence models.

With a validation set, after every epoch :math:`e`:

1. the validation loss :math:`\ell_e` is computed in evaluation mode
   (dropout off);
2. if :math:`\ell_e < \ell^* - \delta`, with :math:`\ell^*` the best loss so
   far and :math:`\delta` = ``min_delta``, the weights are saved as the best;
3. ``ReduceLROnPlateau`` multiplies the learning rate by ``lr_factor`` once
   :math:`\ell_e` has not improved for ``lr_patience`` epochs, down to
   ``min_lr``;
4. training stops once :math:`\ell_e` has not improved for ``patience``
   epochs.

The best weights are restored at the end, and the number of epochs at which
they were reached is reported as ``best_epoch``. Without a validation set,
the model is trained for ``max_epochs`` epochs, optionally following a given
per-epoch learning-rate schedule. This is how a model is refitted on all
training rows with the epoch count and schedule selected on the validation
years.

Determinism: the shuffling generator and the global ``torch`` generator,
which draws the dropout masks, are both seeded from ``seed``. Calling
:func:`seed_everything` before the network is built also fixes its initial
weights, and two runs on the same data and hardware then produce identical
weights.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from vgsales.models.neural.data import make_loader
from vgsales.models.neural.mlp import parameter_groups


def seed_everything(seed: int) -> None:
    """Seed the ``random``, NumPy and ``torch`` global generators."""
    random.seed(seed)
    # For third-party code that still draws from the legacy global state.
    np.random.seed(seed % 2**32)  # noqa: NPY002
    torch.manual_seed(seed)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization settings of :func:`train_model`.

    Attributes
    ----------
    max_epochs : int, default 200
        Upper bound on the number of passes over the training data.
    batch_size : int, default 256
        Rows per mini-batch.
    learning_rate : float, default 1e-3
        Initial AdamW learning rate.
    weight_decay : float, default 1e-5
        Decoupled weight decay of weight matrices and embedding tables.
    patience : int, default 20
        Epochs without validation improvement before training stops.
    min_delta : float, default 0.0
        Minimum decrease of the validation loss counted as an improvement.
    lr_factor : float, default 0.5
        Factor applied to the learning rate on a plateau.
    lr_patience : int, default 5
        Epochs without validation improvement before the rate is reduced.
    min_lr : float, default 1e-5
        Lower bound on the learning rate.
    grad_clip : float or None, default 1.0
        Maximum global gradient norm, or None for no clipping.
    seed : int, default 0
        Seed of the shuffling generator and of the dropout masks.
    """

    max_epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    min_delta: float = 0.0
    lr_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-5
    grad_clip: float | None = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("max_epochs", "batch_size", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.lr_patience < 0:
            raise ValueError("lr_patience must be >= 0")
        if not self.learning_rate > 0 or self.min_lr < 0:
            raise ValueError("learning_rate must be positive and min_lr >= 0")
        if self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("weight_decay and min_delta must be >= 0")
        if not 0 < self.lr_factor < 1:
            raise ValueError("lr_factor must lie in (0, 1)")
        if self.grad_clip is not None and not self.grad_clip > 0:
            raise ValueError("grad_clip must be positive or None")


@dataclass
class TrainingHistory:
    """Per-epoch record of a training run.

    Attributes
    ----------
    train_loss : list of float
        Mean training loss of each epoch, with dropout active.
    val_loss : list of float
        Validation loss after each epoch; empty without a validation set.
    learning_rate : list of float
        Learning rate used in each epoch.
    best_epoch : int
        Number of epochs after which the restored weights were reached; the
        last epoch without a validation set.
    best_val_loss : float
        Validation loss of the restored weights, ``NaN`` without one.
    stopped_early : bool
        Whether training stopped before ``max_epochs``.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = math.nan
    stopped_early: bool = False

    @property
    def n_epochs(self) -> int:
        """Number of epochs run."""
        return len(self.train_loss)


def _set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None,
) -> float:
    model.train()
    total, count = 0.0, 0
    for *inputs, target in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(*inputs), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("the training loss is not finite")
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach()) * len(target)
        count += len(target)
    return total / count


@torch.no_grad()
def evaluate_loss(model: nn.Module, dataset: Dataset, batch_size: int = 1024) -> float:
    """Return the mean squared error of ``model`` on ``dataset`` in eval mode."""
    model.eval()
    total, count = 0.0, 0
    for *inputs, target in make_loader(dataset, batch_size):
        error = model(*inputs) - target
        total += float((error**2).sum())
        count += len(target)
    return total / count


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset, batch_size: int = 1024) -> np.ndarray:
    """Predict every row of ``dataset`` in eval mode.

    The items of ``dataset`` may carry a target as their last tensor; it is
    ignored when the number of tensors exceeds the number of model inputs.

    Returns
    -------
    numpy.ndarray of float64
        One prediction per row, in dataset order.
    """
    model.eval()
    has_target = bool(getattr(dataset, "has_target", False))
    outputs = []
    for batch in make_loader(dataset, batch_size):
        inputs = batch[:-1] if has_target else batch
        outputs.append(model(*inputs).detach().cpu())
    if not outputs:
        return np.zeros(0, dtype=np.float64)
    return torch.cat(outputs).numpy().astype(np.float64)


def train_model(
    model: nn.Module,
    train: Dataset,
    validation: Dataset | None = None,
    config: TrainingConfig | None = None,
    lr_schedule: Sequence[float] | None = None,
) -> TrainingHistory:
    """Train ``model`` in place and return the training history.

    Parameters
    ----------
    model : torch.nn.Module
        Network whose ``forward`` takes the input tensors of a batch.
    train : Dataset
        Training rows; items end with the target tensor.
    validation : Dataset, optional
        Rows for early stopping and learning-rate scheduling.
    config : TrainingConfig, optional
        Optimization settings. Defaults to ``TrainingConfig()``.
    lr_schedule : sequence of float, optional
        Learning rate of each epoch when there is no validation set, for
        example the ``learning_rate`` history of a run with validation. Its
        length sets the number of epochs.

    Returns
    -------
    TrainingHistory
        Losses and learning rates per epoch and the selected epoch.
    """
    config = config or TrainingConfig()
    if lr_schedule is not None:
        if validation is not None:
            raise ValueError("lr_schedule is used only without a validation set")
        if len(lr_schedule) == 0 or min(lr_schedule) <= 0:
            raise ValueError("lr_schedule must hold positive learning rates")
    if len(train) == 0:
        raise ValueError("the training set is empty")

    torch.manual_seed(config.seed)
    loader = make_loader(train, config.batch_size, shuffle=True, seed=config.seed)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.weight_decay), lr=config.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        threshold=config.min_delta,
        threshold_mode="abs",
        min_lr=config.min_lr,
    )
    n_epochs = len(lr_schedule) if lr_schedule is not None else config.max_epochs
    history = TrainingHistory()
    best_state = None
    wait = 0

    for epoch in range(1, n_epochs + 1):
        if lr_schedule is not None:
            _set_learning_rate(optimizer, lr_schedule[epoch - 1])
        history.learning_rate.append(float(optimizer.param_groups[0]["lr"]))
        history.train_loss.append(
            _train_epoch(model, loader, optimizer, config.grad_clip)
        )
        if validation is None:
            continue

        loss = evaluate_loss(model, validation)
        history.val_loss.append(loss)
        if loss < history.best_val_loss - config.min_delta or best_state is None:
            history.best_val_loss, history.best_epoch = loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        scheduler.step(loss)
        if wait >= config.patience:
            history.stopped_early = epoch < n_epochs
            break

    if validation is None:
        history.best_epoch = history.n_epochs
    else:
        model.load_state_dict(best_state)
    model.eval()
    return history
