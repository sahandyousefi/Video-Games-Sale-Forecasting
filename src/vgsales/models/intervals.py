r"""Registry of the prediction interval methods.

Every method returns an unfitted estimator with ``predict`` and
``predict_interval`` that takes the pre-release feature frame and the
modelling-scale target, at the nominal level ``intervals.alpha``, so that
all methods can be cross-validated on identical folds by
:func:`vgsales.evaluation.interval_cv.compare_intervals`.

- ``quantile``: LightGBM quantile regression at the levels
  :math:`\alpha / 2`, :math:`1 / 2` and :math:`1 - \alpha / 2`
  (:mod:`vgsales.models.quantile`).
- ``conformal``: split conformal intervals around ``intervals.point_model``
  (:mod:`vgsales.models.conformal`).
- ``cqr``: conformalized quantile regression, the ``quantile`` models fitted
  on the earlier training years and calibrated on the latest
  ``intervals.calibration_years``.
"""

from __future__ import annotations

from typing import Final

from sklearn.base import BaseEstimator

from vgsales.config import Config
from vgsales.models.conformal import (
    ConformalQuantileRegressor,
    SplitConformalRegressor,
)
from vgsales.models.quantile import make_quantile_interval
from vgsales.models.registry import make_model, missing_dependency

INTERVAL_METHODS: Final = ("quantile", "conformal", "cqr")


def _check_method(method: str) -> None:
    if method not in INTERVAL_METHODS:
        raise ValueError(
            f"unknown interval method {method!r}; choose from {INTERVAL_METHODS}"
        )


def interval_dependency(method: str, config: Config) -> str | None:
    """Return an install hint when ``method`` needs a missing library, else None.

    Raises
    ------
    ValueError
        If ``method`` is unknown or the point model is not registered.
    """
    _check_method(method)
    if method == "conformal":
        return missing_dependency(config.intervals.point_model)
    return missing_dependency("lightgbm")


def make_interval_model(method: str, config: Config) -> BaseEstimator:
    """Build the unfitted interval model of ``method`` from the configuration.

    Parameters
    ----------
    method : str
        Name from :data:`INTERVAL_METHODS`.
    config : Config
        Pipeline configuration; ``intervals`` sets the level and the
        method parameters, ``features`` the preprocessing and ``seed`` the
        random state of the boosting models.

    Returns
    -------
    estimator
        Unfitted interval model.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    ImportError
        If a library the method needs is not installed.
    """
    hint = interval_dependency(method, config)
    if hint is not None:
        raise ImportError(f"interval method {method!r} is unavailable: {hint}")
    settings = config.intervals
    if method == "conformal":
        return SplitConformalRegressor(
            make_model(settings.point_model, config),
            alpha=settings.alpha,
            calibration_years=settings.calibration_years,
        )
    params = {"random_state": config.seed, **settings.quantile_params}
    quantile = make_quantile_interval(settings.alpha, config.features, **params)
    if method == "quantile":
        return quantile
    return ConformalQuantileRegressor(
        quantile, alpha=settings.alpha, calibration_years=settings.calibration_years
    )
