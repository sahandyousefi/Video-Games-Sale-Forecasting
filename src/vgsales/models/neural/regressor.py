r"""Scikit-learn estimator around the entity-embedding MLP.

:class:`EmbeddingMLPRegressor` takes the pre-release feature frame and the
target on the modelling scale, like the other registered models, so it is
cross-validated on the same folds by
:func:`vgsales.evaluation.cv.compare_models`.

The number of epochs is selected with the same temporal inner split as the
boosting models (:mod:`vgsales.models.boosting`). Let :math:`T` be the latest
release year of the training rows and :math:`v` = ``validation_years``. Rows
released after :math:`T - v` form the inner validation fold, and earlier
rows of titles that also appear in it are purged. The encoder
(:class:`vgsales.features.tabular.TabularEncoder`) and the network are fitted
on the remaining rows by :func:`vgsales.models.neural.training.train_model`,
which stops early on the validation loss and returns the best epoch
:math:`e^*` and the learning rate used in each epoch. With ``refit=True`` the
encoder and a newly initialized network are then fitted on all training rows
for :math:`e^*` epochs, replaying the learning rates of the first :math:`e^*`
epochs of the selection run. The outer validation years of a
cross-validation fold therefore never influence :math:`e^*`.

PyTorch is an optional dependency (``pip install "vgsales[dl]"``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin

from vgsales.config import FeatureConfig
from vgsales.data.splits import holdout_split
from vgsales.data.temporal import RELEASE_YEAR
from vgsales.features.prerelease import GROUP_COLUMN
from vgsales.features.tabular import TabularEncoder
from vgsales.models.neural.data import TabularDataset
from vgsales.models.neural.mlp import EntityEmbeddingMLP
from vgsales.models.neural.training import (
    TrainingConfig,
    TrainingHistory,
    predict,
    seed_everything,
    train_model,
)


class EmbeddingMLPRegressor(RegressorMixin, BaseEstimator):
    """Entity-embedding MLP with temporal early stopping.

    Parameters
    ----------
    config : FeatureConfig, optional
        Feature configuration of the encoder.
    hidden : sequence of int, default (128, 64)
        Hidden layer widths.
    dropout : float, default 0.1
        Dropout rate after each hidden layer.
    embedding_dropout : float, default 0.0
        Dropout rate on the concatenated embeddings.
    max_embedding_dim : int, default 16
        Cap on the embedding width.
    learning_rate : float, default 1e-3
        Initial AdamW learning rate.
    weight_decay : float, default 1e-5
        Decoupled weight decay of weight matrices and embedding tables.
    batch_size : int, default 256
        Rows per mini-batch.
    max_epochs : int, default 200
        Maximum number of epochs of the selection run, and the number of
        epochs when early stopping does not run.
    patience : int, default 20
        Epochs without validation improvement before training stops.
    lr_patience : int, default 5
        Epochs without validation improvement before the learning rate is
        multiplied by ``lr_factor``.
    lr_factor : float, default 0.5
        Learning-rate reduction factor.
    validation_years : int, default 2
        Latest training release years that form the inner validation fold.
        Zero disables early stopping.
    refit : bool, default True
        Refit on all training rows for the selected number of epochs. When
        False, the network of the selection run is kept.
    random_state : int, default 0
        Seed of the initial weights, the shuffling and the dropout masks.

    Attributes
    ----------
    encoder_ : TabularEncoder
        Encoder of the kept model.
    network_ : EntityEmbeddingMLP
        Fitted network, in evaluation mode.
    history_ : TrainingHistory
        Training history of the kept network.
    selection_history_ : TrainingHistory or None
        History of the run with the inner validation fold, or None when
        early stopping did not run.
    best_epoch_ : int
        Number of epochs of the kept network.
    early_stopping_years_ : tuple of int or None
        First and last release year of the inner validation fold.
    validation_rmse_ : float
        RMSE on the modelling scale of the inner validation fold at the best
        epoch, ``NaN`` when early stopping did not run.
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        hidden: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
        max_embedding_dim: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        validation_years: int = 2,
        refit: bool = True,
        random_state: int = 0,
    ) -> None:
        self.config = config
        self.hidden = hidden
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.max_embedding_dim = max_embedding_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.validation_years = validation_years
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

    def _inner_split(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        if self.validation_years < 0:
            raise ValueError("validation_years must be >= 0")
        if self.validation_years == 0:
            return None
        groups = X[GROUP_COLUMN] if GROUP_COLUMN in X.columns else None
        try:
            return holdout_split(X[RELEASE_YEAR], self.validation_years, groups)
        except ValueError:
            warnings.warn(
                f"the training rows span too few release years for a "
                f"{self.validation_years}-year validation fold; early stopping "
                f"is skipped and {self.max_epochs} epochs are used",
                UserWarning,
                stacklevel=3,
            )
            return None

    def _fit_network(
        self,
        X: pd.DataFrame,
        z: np.ndarray,
        config: TrainingConfig,
        validation: tuple[pd.DataFrame, np.ndarray] | None = None,
        lr_schedule: Sequence[float] | None = None,
    ) -> tuple[TabularEncoder, EntityEmbeddingMLP, TrainingHistory]:
        encoder = TabularEncoder(self.config).fit(X, z)
        train = TabularDataset(encoder.transform(X), z)
        val = None
        if validation is not None:
            X_val, z_val = validation
            val = TabularDataset(encoder.transform(X_val), z_val)
        seed_everything(self.random_state)
        network = EntityEmbeddingMLP(
            encoder.cardinalities_,
            encoder.n_numeric_,
            hidden=tuple(self.hidden),
            dropout=self.dropout,
            embedding_dropout=self.embedding_dropout,
            max_embedding_dim=self.max_embedding_dim,
        )
        network.init_output_bias(float(z.mean()))
        history = train_model(network, train, val, config, lr_schedule)
        return encoder, network, history

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> EmbeddingMLPRegressor:
        """Select the number of epochs on the latest years and fit the network.

        Parameters
        ----------
        X : pandas.DataFrame
            Pre-release feature frame with the ``release_year`` column and,
            for purging, the ``title_key`` column.
        y : array-like of float
            Target on the modelling scale.

        Returns
        -------
        EmbeddingMLPRegressor
            The fitted estimator.
        """
        if not isinstance(X, pd.DataFrame) or RELEASE_YEAR not in X.columns:
            raise ValueError(f"X must be a DataFrame with a '{RELEASE_YEAR}' column")
        z = np.asarray(y, dtype=np.float64)
        if z.shape != (len(X),) or not np.isfinite(z).all():
            raise ValueError(f"y must hold {len(X)} finite values")
        config = self._training_config()

        inner = self._inner_split(X)
        if inner is None:
            self.selection_history_ = None
            self.early_stopping_years_ = None
            self.validation_rmse_ = float("nan")
            self.encoder_, self.network_, self.history_ = self._fit_network(
                X, z, config
            )
        else:
            train_idx, val_idx = inner
            validation = (X.iloc[val_idx], z[val_idx])
            fitted = self._fit_network(
                X.iloc[train_idx], z[train_idx], config, validation
            )
            history = fitted[2]
            years = X[RELEASE_YEAR].to_numpy()[val_idx]
            self.selection_history_ = history
            self.early_stopping_years_ = (int(years.min()), int(years.max()))
            self.validation_rmse_ = math.sqrt(history.best_val_loss)
            if self.refit:
                schedule = history.learning_rate[: history.best_epoch]
                fitted = self._fit_network(X, z, config, lr_schedule=schedule)
            self.encoder_, self.network_, self.history_ = fitted

        self.best_epoch_ = self.history_.best_epoch
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target on the modelling scale."""
        arrays = self.encoder_.transform(X)
        return predict(self.network_, TabularDataset(arrays), self.batch_size)
