"""Fit on the modelling scale of the target, predict on the original scale."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor

from vgsales.config import TARGET_TRANSFORMS


def with_target_transform(estimator: BaseEstimator, transform: str) -> BaseEstimator:
    r"""Wrap a regressor so that it learns the transformed target.

    Parameters
    ----------
    estimator : estimator
        Unfitted scikit-learn regressor or pipeline.
    transform : {"log1p", "identity"}
        Transform from ``target.transform`` in the configuration.

    Returns
    -------
    estimator
        For ``"log1p"``, a ``TransformedTargetRegressor`` that fits
        ``estimator`` on :math:`\log(1 + y)` and returns
        :math:`\exp(\hat z) - 1`. For ``"identity"``, ``estimator`` itself.
        In both cases :meth:`fit` takes and :meth:`predict` returns sales on
        the original scale.

    Raises
    ------
    ValueError
        If ``transform`` is not supported.
    """
    if transform == "identity":
        return estimator
    if transform == "log1p":
        return TransformedTargetRegressor(
            regressor=estimator, func=np.log1p, inverse_func=np.expm1
        )
    raise ValueError(f"transform must be one of {TARGET_TRANSFORMS}, got {transform!r}")
