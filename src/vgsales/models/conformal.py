r"""Split conformal prediction intervals calibrated on later release years.

Let :math:`z = \log(1 + y)` be the modelling-scale target and :math:`t_i`
the release year of row :math:`i`. The training rows of a fold are divided
by release year into a proper training set :math:`\mathcal{D}_1` and a
calibration set :math:`\mathcal{D}_2` that holds the latest ``c`` years
(``calibration_years``). Rows of :math:`\mathcal{D}_1` whose title also has
rows in :math:`\mathcal{D}_2` are dropped, as for the holdout split, so that
no title informs both the model and its calibration.

A point model :math:`\hat\mu` is fitted on :math:`\mathcal{D}_1` only. The
nonconformity score of a calibration row is the absolute residual
:math:`s_i = |z_i - \hat\mu(x_i)|`. With :math:`n = |\mathcal{D}_2|` and the
order statistics :math:`s_{(1)} \le \dots \le s_{(n)}`, the conformal
quantile is

.. math::

    \hat q = s_{(k)}, \qquad k = \lceil (n + 1)(1 - \alpha) \rceil,

and the interval of a new row is :math:`[\hat\mu(x) - \hat q,
\hat\mu(x) + \hat q]` on the modelling scale. If the calibration rows and
the new row are exchangeable, the interval covers :math:`z` with probability
at least :math:`1 - \alpha`, and at most :math:`1 - \alpha + 1 / (n + 1)`
when the scores have no ties (Vovk et al., 2005; Lei et al., 2018). The
level requires :math:`k \le n`, that is :math:`n \ge 1 / \alpha - 1`
calibration rows.

On the original scale the interval is
:math:`[(1 + \hat y) e^{-\hat q} - 1, (1 + \hat y) e^{\hat q} - 1]` with
:math:`\hat y = e^{\hat\mu(x)} - 1`: a band of constant relative width
around the point forecast, whose lower bound is clipped at zero sales when
it is scored.

Release years are not exchangeable. Taking the calibration set from the
latest years of the training window places it as close as possible to the
years being forecast, but under drift in the sales distribution the
guarantee does not hold, and coverage on later years is measured
empirically. The point model is not refitted on :math:`\mathcal{D}_1 \cup
\mathcal{D}_2`, since that would make the calibration residuals in-sample
and void the guarantee.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from vgsales.data.splits import holdout_split
from vgsales.data.temporal import RELEASE_YEAR
from vgsales.features.prerelease import GROUP_COLUMN


def _check_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha}")
    return float(alpha)


def min_calibration_size(alpha: float) -> int:
    """Smallest calibration set for which the conformal quantile is finite."""
    alpha = _check_alpha(alpha)
    return math.ceil(round(1.0 / alpha - 1.0, 9))


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Finite-sample conformal quantile of calibration scores.

    Parameters
    ----------
    scores : array-like of float
        Nonconformity scores of the calibration rows.
    alpha : float
        Nominal miscoverage in (0, 1).

    Returns
    -------
    float
        The :math:`\\lceil (n + 1)(1 - \\alpha) \\rceil`-th smallest score.

    Raises
    ------
    ValueError
        If the scores are empty or not finite, ``alpha`` lies outside
        (0, 1), or there are fewer than :func:`min_calibration_size` scores.
    """
    alpha = _check_alpha(alpha)
    values = np.sort(np.asarray(scores, dtype=np.float64))
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty one-dimensional finite array")
    n = values.size
    # Rounding guards against products such as 20 * 0.9 landing just above 18.
    rank = math.ceil(round((n + 1) * (1.0 - alpha), 9))
    if rank > n:
        raise ValueError(
            f"{n} calibration rows are too few for alpha={alpha}; at least "
            f"{min_calibration_size(alpha)} are needed"
        )
    return float(values[rank - 1])


def calibration_split(
    X: pd.DataFrame, calibration_years: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows into a proper training set and a calibration set.

    Parameters
    ----------
    X : pandas.DataFrame
        Training rows with the ``release_year`` column and, for purging, the
        ``title_key`` column.
    calibration_years : int
        Number of latest release years that form the calibration set.

    Returns
    -------
    train_idx, calibration_idx : numpy.ndarray
        Positional indices.

    Raises
    ------
    ValueError
        If ``X`` lacks the release year or spans too few years.
    """
    if not isinstance(X, pd.DataFrame) or RELEASE_YEAR not in X.columns:
        raise ValueError(f"X must be a DataFrame with a '{RELEASE_YEAR}' column")
    if isinstance(calibration_years, bool) or not isinstance(calibration_years, int):
        raise TypeError("calibration_years must be an integer")
    if calibration_years < 1:
        raise ValueError("calibration_years must be >= 1")
    groups = X[GROUP_COLUMN] if GROUP_COLUMN in X.columns else None
    try:
        return holdout_split(X[RELEASE_YEAR], calibration_years, groups)
    except ValueError as error:
        raise ValueError(
            f"cannot draw a {calibration_years}-year calibration set: {error}"
        ) from error


class SplitConformalRegressor(RegressorMixin, BaseEstimator):
    """Point model with split conformal intervals from the latest years.

    Parameters
    ----------
    estimator : estimator
        Unfitted point regressor that takes the pre-release feature frame
        and the modelling-scale target, for example a model from
        :func:`vgsales.models.registry.make_model`.
    alpha : float, default 0.1
        Nominal miscoverage of the interval.
    calibration_years : int, default 2
        Number of latest release years of the training rows that form the
        calibration set.

    Attributes
    ----------
    estimator_ : estimator
        Point model fitted on the proper training set.
    scores_ : numpy.ndarray
        Absolute residuals of the calibration rows.
    quantile_ : float
        Conformal quantile :math:`\\hat q` of the scores.
    calibration_years_ : tuple of int
        First and last release year of the calibration set.
    n_train_, n_calibration_ : int
        Sizes of the proper training set and of the calibration set.
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        alpha: float = 0.1,
        calibration_years: int = 2,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_years = calibration_years

    def _fit_point_model(
        self, X: pd.DataFrame, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.estimator is None:
            raise ValueError("a point estimator is required")
        _check_alpha(self.alpha)
        train_idx, cal_idx = calibration_split(X, self.calibration_years)
        z = np.asarray(y, dtype=np.float64)
        if z.shape != (len(X),) or not np.isfinite(z).all():
            raise ValueError(f"y must hold {len(X)} finite values")
        self.estimator_ = clone(self.estimator).fit(X.iloc[train_idx], z[train_idx])
        years = X[RELEASE_YEAR].to_numpy()[cal_idx]
        self.calibration_years_ = (int(years.min()), int(years.max()))
        self.n_train_ = int(train_idx.size)
        self.n_calibration_ = int(cal_idx.size)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return z, cal_idx

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> SplitConformalRegressor:
        """Fit the point model and calibrate the interval half-width.

        Parameters
        ----------
        X : pandas.DataFrame
            Pre-release feature frame with the ``release_year`` column.
        y : array-like of float
            Target on the modelling scale.

        Returns
        -------
        SplitConformalRegressor
            The fitted estimator.
        """
        z, cal_idx = self._fit_point_model(X, y)
        residuals = z[cal_idx] - self.estimator_.predict(X.iloc[cal_idx])
        self.scores_ = np.abs(np.asarray(residuals, dtype=np.float64))
        self.quantile_ = conformal_quantile(self.scores_, self.alpha)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target on the modelling scale."""
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict(X), dtype=np.float64)

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds of the conformal interval on the modelling scale."""
        check_is_fitted(self, "quantile_")
        center = self.predict(X)
        return center - self.quantile_, center + self.quantile_
