r"""GRU forecaster for the monthly aggregate series.

The network reads a window of :math:`L` months of one segment, as built by
:func:`vgsales.features.windows.make_windows`, and predicts the next
:math:`H` standardized values at once. With the input vectors
:math:`x_{t-L+1}, \dots, x_t \in \mathbb{R}^3` of a window, a gated recurrent
unit (Cho et al., 2014) updates its hidden state :math:`h_s \in
\mathbb{R}^{p}` as

.. math::

    r_s &= \sigma(W_r x_s + U_r h_{s-1} + b_r), \\
    a_s &= \sigma(W_a x_s + U_a h_{s-1} + b_a), \\
    \tilde h_s &= \tanh\big(W_h x_s + U_h (r_s \odot h_{s-1}) + b_h\big), \\
    h_s &= (1 - a_s) \odot \tilde h_s + a_s \odot h_{s-1},

with reset gate :math:`r_s`, update gate :math:`a_s` and :math:`h_{t-L} = 0`.
The forecast is a linear map of the final state and a learned embedding
:math:`e_k \in \mathbb{R}^{q}` of the segment,

.. math::

    (\hat u_{k,t+1}, \dots, \hat u_{k,t+H}) = V \,[h_t, e_k] + c,

so all segments share the recurrent weights while each keeps its own offset
and seasonal adjustment. Stacked layers pass their state sequence to the
next layer, with dropout between layers.

:class:`GRUForecaster` fits the network on all windows of a wide series with
the procedure of :class:`vgsales.models.neural.regressor.EmbeddingMLPRegressor`.
The last ``validation_months`` months form an inner validation period: the
scaler is fitted on the months before it, training windows end their targets
before it and validation windows start their targets in it
(:func:`vgsales.features.windows.split_origins`). The selected number of
epochs is then used to refit the scaler and a newly initialized network on
all months. Forecasts are mapped back to units with
:meth:`vgsales.features.windows.SeriesScaler.inverse_transform`.

PyTorch is an optional dependency (``pip install "vgsales[dl]"``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from vgsales.features.windows import (
    N_CHANNELS,
    SequenceWindows,
    SeriesScaler,
    last_windows,
    make_windows,
    split_origins,
)
from vgsales.models.neural.training import (
    TrainingConfig,
    TrainingHistory,
    predict,
    seed_everything,
    train_model,
)


class SequenceDataset(Dataset):
    """Windows as tensors: inputs, segment index and an optional target.

    Indexing with a sequence of positions returns stacked tensors, so the
    dataset works with :func:`vgsales.models.neural.data.make_loader`.
    """

    def __init__(self, windows: SequenceWindows) -> None:
        self.inputs = torch.as_tensor(windows.inputs, dtype=torch.float32)
        self.segment = torch.as_tensor(windows.segment, dtype=torch.long)
        self.target = (
            None
            if windows.target is None
            else torch.as_tensor(windows.target, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def has_target(self) -> bool:
        """Whether the items carry the target."""
        return self.target is not None

    def __getitem__(
        self, index: int | Sequence[int] | torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        if self.target is None:
            return self.inputs[index], self.segment[index]
        return self.inputs[index], self.segment[index], self.target[index]


class GRUNetwork(nn.Module):
    """Multi-horizon GRU with a segment embedding.

    Parameters
    ----------
    n_segments : int
        Number of segments, the size of the embedding table.
    horizon : int
        Number of forecast months :math:`H`.
    hidden_size : int, default 32
        Width :math:`p` of the hidden state.
    num_layers : int, default 1
        Number of stacked GRU layers.
    dropout : float, default 0.0
        Dropout between stacked layers and on the input of the output layer.
    segment_dim : int, default 4
        Width :math:`q` of the segment embedding; zero removes it.
    n_channels : int, default 3
        Number of inputs per month.
    """

    def __init__(
        self,
        n_segments: int,
        horizon: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        segment_dim: int = 4,
        n_channels: int = N_CHANNELS,
    ) -> None:
        super().__init__()
        if n_segments < 1 or horizon < 1 or n_channels < 1:
            raise ValueError("n_segments, horizon and n_channels must be >= 1")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be >= 1")
        if segment_dim < 0:
            raise ValueError("segment_dim must be >= 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.horizon = int(horizon)
        self.gru = nn.GRU(
            n_channels,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.segment_embedding = (
            nn.Embedding(n_segments, segment_dim) if segment_dim > 0 else None
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + segment_dim, horizon)

    def forward(self, inputs: torch.Tensor, segment: torch.Tensor) -> torch.Tensor:
        """Return forecasts of shape ``(batch, horizon)``."""
        _, state = self.gru(inputs)
        features = state[-1]
        if self.segment_embedding is not None:
            features = torch.cat([features, self.segment_embedding(segment)], dim=1)
        return self.head(self.dropout(features))


class GRUForecaster:
    """Fit a :class:`GRUNetwork` on wide monthly series and forecast them.

    Parameters
    ----------
    input_length : int, default 24
        Input months :math:`L` of a window.
    horizon : int, default 12
        Forecast months :math:`H`.
    hidden_size, num_layers, dropout, segment_dim
        Architecture; see :class:`GRUNetwork`.
    learning_rate : float, default 1e-3
        Initial AdamW learning rate.
    weight_decay : float, default 1e-4
        Decoupled weight decay of weight matrices and embedding tables.
    batch_size : int, default 64
        Windows per mini-batch.
    max_epochs : int, default 200
        Maximum number of epochs of the selection run.
    patience, lr_patience, lr_factor
        Early stopping and plateau schedule; see
        :class:`vgsales.models.neural.training.TrainingConfig`.
    validation_months : int, default 24
        Length of the inner validation period, at least ``horizon``. Zero
        disables early stopping and trains for ``max_epochs`` epochs.
    refit : bool, default True
        Refit on all months for the selected number of epochs.
    random_state : int, default 0
        Seed of the initial weights, the shuffling and the dropout masks.

    Attributes
    ----------
    scaler_ : SeriesScaler
        Scaler of the kept network.
    network_ : GRUNetwork
        Fitted network, in evaluation mode.
    history_, selection_history_ : TrainingHistory
        Histories of the kept network and of the selection run (None when
        early stopping did not run).
    best_epoch_ : int
        Number of epochs of the kept network.
    validation_rmse_ : float
        RMSE of the standardized validation targets at the best epoch,
        ``NaN`` without early stopping.
    segments_ : pandas.Index
        Segment labels in fitted order.
    end_ : pandas.Timestamp
        Last month of the fitting series.
    """

    def __init__(
        self,
        input_length: int = 24,
        horizon: int = 12,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        segment_dim: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        validation_months: int = 24,
        refit: bool = True,
        random_state: int = 0,
    ) -> None:
        self.input_length = input_length
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.segment_dim = segment_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.validation_months = validation_months
        self.refit = refit
        self.random_state = random_state

    def _training_config(self) -> TrainingConfig:
        return TrainingConfig(
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            lr_factor=self.lr_factor,
            lr_patience=self.lr_patience,
            seed=self.random_state,
        )

    def _new_network(self, n_segments: int) -> GRUNetwork:
        seed_everything(self.random_state)
        return GRUNetwork(
            n_segments,
            self.horizon,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            segment_dim=self.segment_dim,
        )

    def _fit_network(
        self,
        wide: pd.DataFrame,
        scaler_months: int,
        config: TrainingConfig,
        split: tuple[pd.Timestamp, pd.Timestamp] | None = None,
        lr_schedule: Sequence[float] | None = None,
    ) -> tuple[SeriesScaler, GRUNetwork, TrainingHistory]:
        scaler = SeriesScaler().fit(wide.iloc[:scaler_months])
        windows = make_windows(scaler.transform(wide), self.input_length, self.horizon)
        train, validation = windows, None
        if split is not None:
            last_train, first_val = split
            train = windows.subset(windows.origin <= last_train)
            validation = SequenceDataset(windows.subset(windows.origin >= first_val))
        network = self._new_network(wide.shape[1])
        history = train_model(
            network, SequenceDataset(train), validation, config, lr_schedule
        )
        return scaler, network, history

    def fit(self, wide: pd.DataFrame) -> GRUForecaster:
        """Select the number of epochs on the last months and fit the network.

        Parameters
        ----------
        wide : pandas.DataFrame
            Monthly sales in units, a complete monthly grid as index and one
            column per segment, as returned by
            :func:`vgsales.data.series.to_wide`.

        Returns
        -------
        GRUForecaster
            The fitted forecaster.
        """
        if self.validation_months < 0:
            raise ValueError("validation_months must be >= 0")
        config = self._training_config()
        n_months = len(wide)
        self.selection_history_ = None
        self.validation_rmse_ = float("nan")
        if self.validation_months == 0:
            fitted = self._fit_network(wide, n_months, config)
        else:
            split = split_origins(
                pd.DatetimeIndex(wide.index), self.horizon, self.validation_months
            )
            fitted = self._fit_network(
                wide, n_months - self.validation_months, config, split
            )
            history = fitted[2]
            self.selection_history_ = history
            self.validation_rmse_ = math.sqrt(history.best_val_loss)
            if self.refit:
                schedule = history.learning_rate[: history.best_epoch]
                fitted = self._fit_network(wide, n_months, config, lr_schedule=schedule)
        self.scaler_, self.network_, self.history_ = fitted
        self.best_epoch_ = self.history_.best_epoch
        self.segments_ = pd.Index(wide.columns)
        self.end_ = pd.Timestamp(wide.index[-1])
        self._history = wide.iloc[-self.input_length :]
        return self

    def forecast(self, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """Forecast the ``horizon`` months after the end of ``history``.

        Parameters
        ----------
        history : pandas.DataFrame, optional
            Wide monthly series in units with at least ``input_length``
            months and the fitted segments. Defaults to the end of the
            fitting series. A later history gives forecasts from a later
            origin without refitting.

        Returns
        -------
        pandas.DataFrame
            Forecast sales in units, the forecast months as index and the
            fitted segments as columns.
        """
        if not hasattr(self, "network_"):
            raise RuntimeError("the forecaster is not fitted")
        history = self._history if history is None else history
        windows = last_windows(self.scaler_.transform(history), self.input_length)
        scaled = predict(self.network_, SequenceDataset(windows), self.batch_size)
        values = self.scaler_.inverse_transform(scaled, segment=windows.segment)
        start = pd.Timestamp(history.index[-1]) + pd.DateOffset(months=1)
        index = pd.date_range(start, periods=self.horizon, freq="MS", name="period")
        return pd.DataFrame(values.T, index=index, columns=self.segments_)
