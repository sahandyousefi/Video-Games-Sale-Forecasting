r"""LightGBM quantile regression for central prediction intervals.

For a nominal miscoverage :math:`\alpha`, three conditional quantiles of the
modelling-scale target :math:`z = \log(1 + y)` are estimated at the levels
:math:`\tau \in \{\alpha / 2, 1 / 2, 1 - \alpha / 2\}`. Each level gets its
own LightGBM model (Ke et al., 2017) trained on the pinball loss
:math:`\rho_\tau` (Koenker and Bassett, 1978), with the preprocessing and
native categorical splits of :class:`vgsales.models.boosting.LightGBMRegressor`.
The number of rounds of each model is selected by temporal early stopping on
the pinball loss at its own level, evaluated on the latest
``validation_years`` of the training rows, so the outer validation years of a
cross-validation fold never influence it.

Quantiles estimated separately can cross. The three predictions of each row
are sorted, the rearrangement of Chernozhukov et al. (2010), which yields
monotone quantile estimates and does not increase their :math:`L^p`
distance to the true quantile function. The median is the point prediction,
and :math:`[\hat q_{\alpha/2}, \hat q_{1-\alpha/2}]` is the interval. Because
quantiles are equivariant under the monotone map :math:`z \mapsto e^z - 1`,
the bounds back-transformed to the original scale are quantiles of the
sales :math:`y` at the same levels.

Quantile regression targets nominal coverage only asymptotically and under
a correctly specified model; the coverage on later release years is checked
empirically and can be corrected by conformal calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor

from vgsales.config import FeatureConfig
from vgsales.data.temporal import RELEASE_YEAR
from vgsales.models.boosting import WRAPPER_PARAMS, LightGBMRegressor


def _check_level(value: float, name: str) -> float:
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")
    return float(value)


def interval_levels(alpha: float) -> tuple[float, float, float]:
    """Quantile levels of the lower bound, the median and the upper bound.

    Raises
    ------
    ValueError
        If ``alpha`` lies outside (0, 1).
    """
    alpha = _check_level(alpha, "alpha")
    return (alpha / 2, 0.5, 1 - alpha / 2)


def rearrange_quantiles(quantiles: ArrayLike) -> np.ndarray:
    """Sort the predicted quantiles of each row into increasing order.

    Parameters
    ----------
    quantiles : array-like of shape (n_rows, n_levels)
        Predictions for increasing quantile levels.

    Returns
    -------
    numpy.ndarray
        Row-wise sorted copy.
    """
    values = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("quantiles must be a two-dimensional array")
    return np.sort(values, axis=1)


class LightGBMQuantileRegressor(LightGBMRegressor):
    """LightGBM on the pinball loss at one quantile level.

    Parameters
    ----------
    quantile : float, default 0.5
        Quantile level :math:`\\tau` in (0, 1). The training objective and
        the early-stopping metric are the pinball loss at this level.
    config, n_estimators, learning_rate, early_stopping_rounds,
    validation_years, refit, random_state, params
        As for :class:`vgsales.models.boosting.LightGBMRegressor`.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        config: FeatureConfig | None = None,
        n_estimators: int = 2000,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 100,
        validation_years: int = 2,
        refit: bool = True,
        random_state: int = 0,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            validation_years=validation_years,
            refit=refit,
            random_state=random_state,
            params=params,
        )
        self.quantile = quantile

    def _check_params(self) -> None:
        super()._check_params()
        _check_level(self.quantile, "quantile")

    def _objective(self) -> dict[str, Any]:
        return {
            "objective": "quantile",
            "alpha": float(self.quantile),
            "metric": "quantile",
        }


class QuantileIntervalRegressor(RegressorMixin, BaseEstimator):
    """Median and central interval from three LightGBM quantile models.

    Parameters
    ----------
    alpha : float, default 0.1
        Nominal miscoverage of the central interval.
    config : FeatureConfig, optional
        Feature configuration of the preprocessing pipeline.
    n_estimators, learning_rate, early_stopping_rounds, validation_years,
    refit, random_state, params
        Passed to every :class:`LightGBMQuantileRegressor`.

    Attributes
    ----------
    levels_ : tuple of float
        Quantile levels of the lower bound, the median and the upper bound.
    models_ : list of LightGBMQuantileRegressor
        Fitted model of each level.
    n_estimators_ : tuple of int
        Selected number of rounds of each model.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        config: FeatureConfig | None = None,
        n_estimators: int = 2000,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 100,
        validation_years: int = 2,
        refit: bool = True,
        random_state: int = 0,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self.alpha = alpha
        self.config = config
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_years = validation_years
        self.refit = refit
        self.random_state = random_state
        self.params = params

    def _base_model(self) -> LightGBMQuantileRegressor:
        return LightGBMQuantileRegressor(
            config=self.config,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            validation_years=self.validation_years,
            refit=self.refit,
            random_state=self.random_state,
            params=self.params,
        )

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> QuantileIntervalRegressor:
        """Fit one quantile model per level on the modelling-scale target.

        Parameters
        ----------
        X : pandas.DataFrame
            Pre-release feature frame with the ``release_year`` column.
        y : array-like of float
            Target on the modelling scale.

        Returns
        -------
        QuantileIntervalRegressor
            The fitted estimator.
        """
        if not isinstance(X, pd.DataFrame) or RELEASE_YEAR not in X.columns:
            raise ValueError(f"X must be a DataFrame with a '{RELEASE_YEAR}' column")
        self.levels_ = interval_levels(self.alpha)
        base = self._base_model()
        self.models_ = [
            clone(base).set_params(quantile=level).fit(X, y) for level in self.levels_
        ]
        self.n_estimators_ = tuple(model.n_estimators_ for model in self.models_)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Rearranged quantile predictions on the modelling scale.

        Returns
        -------
        numpy.ndarray of shape (n_rows, 3)
            Lower bound, median and upper bound of each row.
        """
        columns = [model.predict(X) for model in self.models_]
        return rearrange_quantiles(np.column_stack(columns))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the conditional median on the modelling scale."""
        return self.predict_quantiles(X)[:, 1]

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds of the central interval on the modelling scale."""
        quantiles = self.predict_quantiles(X)
        return quantiles[:, 0], quantiles[:, 2]


def make_quantile_interval(
    alpha: float = 0.1, config: FeatureConfig | None = None, **params: Any
) -> QuantileIntervalRegressor:
    """Build an unfitted quantile interval model.

    Parameters
    ----------
    alpha : float, default 0.1
        Nominal miscoverage of the central interval.
    config : FeatureConfig, optional
        Feature configuration of the preprocessing pipeline.
    **params
        Arguments listed in :data:`vgsales.models.boosting.WRAPPER_PARAMS`;
        every other keyword is passed to the LightGBM boosters, as for
        :func:`vgsales.models.boosting.make_boosting`.

    Returns
    -------
    QuantileIntervalRegressor
        Unfitted estimator.
    """
    wrapper = {key: params.pop(key) for key in WRAPPER_PARAMS if key in params}
    return QuantileIntervalRegressor(
        alpha=alpha, config=config, params=params or None, **wrapper
    )


def predict_sales_interval(
    model: BaseEstimator, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Interval bounds on the original scale from a fitted interval model.

    Parameters
    ----------
    model : estimator
        Fitted estimator with ``predict_interval``, either bare (fitted on
        sales) or wrapped by
        :func:`vgsales.models.target.with_target_transform`.
    X : pandas.DataFrame
        Feature frame.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper bounds in millions of units.

    Raises
    ------
    TypeError
        If the model does not predict intervals.
    """
    if isinstance(model, TransformedTargetRegressor):
        inner, inverse = model.regressor_, model.inverse_func
    else:
        inner, inverse = model, None
    if not hasattr(inner, "predict_interval"):
        raise TypeError(f"{type(inner).__name__} does not predict intervals")
    lower, upper = inner.predict_interval(X)
    if inverse is not None:
        lower, upper = inverse(lower), inverse(upper)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
