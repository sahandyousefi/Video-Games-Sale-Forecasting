"""Registry of the models that the command line can compare and train.

Every model is built from its name and the pipeline configuration, takes the
pre-release feature frame as input and is fitted on the modelling scale of
the target, so that any two models can be cross-validated on identical folds
by :func:`vgsales.evaluation.cv.compare_models`. Models whose library is an
optional dependency name the module they need, and :func:`is_available`
reports whether it can be imported.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Final

from sklearn.base import BaseEstimator

from vgsales.config import Config
from vgsales.models.baselines import make_baseline
from vgsales.models.boosting import make_boosting


@dataclass(frozen=True)
class ModelSpec:
    """Description of a registered model.

    Attributes
    ----------
    name : str
        Name used in the configuration and on the command line.
    family : {"baseline", "boosting"}
        Factory that builds the model.
    requires : str or None
        Module of an optional dependency, or None for core dependencies only.
    extra : str or None
        Optional-dependency group of ``pyproject.toml`` that installs it.
    summary : str
        One-line description.
    """

    name: str
    family: str
    requires: str | None
    extra: str | None
    summary: str


REGISTRY: Final[dict[str, ModelSpec]] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "global_median",
            "baseline",
            None,
            None,
            "median of the training target",
        ),
        ModelSpec(
            "group_median",
            "baseline",
            None,
            None,
            "median by genre and console with back-off",
        ),
        ModelSpec(
            "ridge",
            "baseline",
            None,
            None,
            "ridge regression on the standardized features",
        ),
        ModelSpec(
            "xgboost",
            "boosting",
            "xgboost",
            "gbm",
            "XGBoost on one-hot and target-encoded features",
        ),
        ModelSpec(
            "lightgbm",
            "boosting",
            "lightgbm",
            "gbm",
            "LightGBM with native categorical splits",
        ),
        ModelSpec(
            "catboost",
            "boosting",
            "catboost",
            "gbm",
            "CatBoost with ordered target statistics in release order",
        ),
    )
}

MODEL_NAMES: Final = tuple(REGISTRY)


def get_spec(name: str) -> ModelSpec:
    """Return the specification of model ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}") from None


def missing_dependency(name: str) -> str | None:
    """Return an install hint when the library of ``name`` is missing, else None."""
    spec = get_spec(name)
    if spec.requires is None or importlib.util.find_spec(spec.requires) is not None:
        return None
    return f"{spec.requires} is not installed (pip install 'vgsales[{spec.extra}]')"


def is_available(name: str) -> bool:
    """Return whether model ``name`` can be built and fitted."""
    return missing_dependency(name) is None


def model_params(name: str, config: Config, **overrides: Any) -> dict[str, Any]:
    """Resolve the keyword arguments with which model ``name`` is built.

    Configured parameters are updated with ``overrides``. Boosting models
    receive ``config.seed`` as ``random_state`` unless one is given.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    spec = get_spec(name)
    params = {**config.models.params_for(name), **overrides}
    if spec.family == "boosting":
        params.setdefault("random_state", config.seed)
    return params


def make_model(name: str, config: Config, **overrides: Any) -> BaseEstimator:
    """Build the unfitted model ``name`` with its configured parameters.

    Parameters
    ----------
    name : str
        Registered model name.
    config : Config
        Pipeline configuration. ``features`` configures the preprocessing and
        the parameters are resolved by :func:`model_params`.
    **overrides
        Keyword arguments that replace the configured ones.

    Returns
    -------
    estimator
        Unfitted estimator that takes the pre-release feature frame.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    ImportError
        If the optional library of the model is not installed.
    """
    params = model_params(name, config, **overrides)
    hint = missing_dependency(name)
    if hint is not None:
        raise ImportError(f"model {name!r} is unavailable: {hint}")
    if get_spec(name).family == "boosting":
        return make_boosting(name, config.features, **params)
    return make_baseline(name, config.features, **params)
