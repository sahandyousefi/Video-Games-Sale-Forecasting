r"""Search spaces for the gradient-boosting models.

Each space maps a keyword argument of :func:`vgsales.models.registry.make_model`
to a distribution. The number of boosting rounds is not searched: every
trial keeps the configured maximum ``n_estimators`` and selects the number of
rounds by temporal early stopping (:mod:`vgsales.models.boosting`), so the
learning rate is searched jointly with the tree parameters and the selected
round count adapts to it.

Scale parameters that span orders of magnitude (learning rate, regularization
strengths, minimum leaf sizes, number of leaves) are sampled on a log scale,
so that a sampler such as the tree-structured Parzen estimator (Bergstra et
al., 2011) used by Optuna (Akiba et al., 2019) spends comparable effort on
0.01 to 0.1 and on 0.1 to 1. Bounds are wide enough to contain the library
defaults and the configured values of ``configs/default.yaml``:

- XGBoost (Chen and Guestrin, 2016): tree depth, minimum Hessian weight of a
  child, row and column subsampling, and the :math:`\ell_2` and :math:`\ell_1`
  penalties on leaf weights.
- LightGBM (Ke et al., 2017): number of leaves of the leaf-wise trees,
  minimum rows per leaf, subsampling, :math:`\ell_2` penalty, and the minimum
  rows per category group and smoothing of the native categorical splits.
- CatBoost (Prokhorenkova et al., 2018): depth of the oblivious trees,
  :math:`\ell_2` penalty, the random strength that perturbs split scores, and
  the row subsampling rate.

The spaces are plain data. :func:`suggest_params` draws one configuration
from an Optuna trial and is the only function here that needs Optuna.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True)
class FloatParam:
    """Real-valued parameter on ``[low, high]``, optionally log-uniform."""

    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError("low must be smaller than high")
        if self.log and self.low <= 0:
            raise ValueError("a log-scale parameter needs low > 0")

    def contains(self, value: Any) -> bool:
        """Whether ``value`` lies inside the bounds."""
        return isinstance(value, float | int) and self.low <= value <= self.high


@dataclass(frozen=True)
class IntParam:
    """Integer parameter on ``{low, ..., high}``, optionally log-uniform."""

    low: int
    high: int
    log: bool = False

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError("low must be smaller than high")
        if self.log and self.low < 1:
            raise ValueError("a log-scale parameter needs low >= 1")

    def contains(self, value: Any) -> bool:
        """Whether ``value`` is an integer inside the bounds."""
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return self.low <= value <= self.high


Param = FloatParam | IntParam

SearchSpace = Mapping[str, Param]

_LEARNING_RATE: Final = FloatParam(0.01, 0.3, log=True)

SEARCH_SPACES: Final[dict[str, dict[str, Param]]] = {
    "xgboost": {
        "learning_rate": _LEARNING_RATE,
        "max_depth": IntParam(3, 10),
        "min_child_weight": FloatParam(1.0, 100.0, log=True),
        "subsample": FloatParam(0.5, 1.0),
        "colsample_bytree": FloatParam(0.4, 1.0),
        "reg_lambda": FloatParam(1e-3, 100.0, log=True),
        "reg_alpha": FloatParam(1e-3, 10.0, log=True),
    },
    "lightgbm": {
        "learning_rate": _LEARNING_RATE,
        "num_leaves": IntParam(8, 256, log=True),
        "min_child_samples": IntParam(5, 200, log=True),
        "subsample": FloatParam(0.5, 1.0),
        "colsample_bytree": FloatParam(0.4, 1.0),
        "reg_lambda": FloatParam(1e-3, 100.0, log=True),
        "min_data_per_group": IntParam(10, 200, log=True),
        "cat_smooth": FloatParam(1.0, 100.0, log=True),
    },
    "catboost": {
        "learning_rate": _LEARNING_RATE,
        "depth": IntParam(4, 10),
        "l2_leaf_reg": FloatParam(0.5, 50.0, log=True),
        "random_strength": FloatParam(0.01, 10.0, log=True),
        "subsample": FloatParam(0.5, 1.0),
    },
}


def get_space(name: str) -> dict[str, Param]:
    """Return a copy of the search space of model ``name``.

    Raises
    ------
    ValueError
        If no space is defined for ``name``.
    """
    if name not in SEARCH_SPACES:
        raise ValueError(
            f"no search space for {name!r}; spaces exist for {tuple(SEARCH_SPACES)}"
        )
    return dict(SEARCH_SPACES[name])


def suggest_params(trial: Any, space: SearchSpace) -> dict[str, Any]:
    """Draw one configuration of ``space`` from an Optuna trial.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial that records the suggestions; a ``FixedTrial`` replays given
        values.
    space : mapping of str to Param
        Search space, for example ``get_space("xgboost")``.

    Returns
    -------
    dict
        Keyword arguments for :func:`vgsales.models.registry.make_model`.
    """
    params: dict[str, Any] = {}
    for name, param in space.items():
        if isinstance(param, IntParam):
            params[name] = trial.suggest_int(name, param.low, param.high, log=param.log)
        elif isinstance(param, FloatParam):
            params[name] = trial.suggest_float(
                name, param.low, param.high, log=param.log
            )
        else:
            raise TypeError(f"unsupported parameter type for {name}: {param!r}")
    return params


def check_params(name: str, params: Mapping[str, Any]) -> None:
    """Check that ``params`` names searched parameters with in-range values.

    Used to validate tuned parameters read back from disk.

    Raises
    ------
    ValueError
        If a parameter is not in the space of ``name`` or out of range.
    """
    space = get_space(name)
    unknown = sorted(set(params) - set(space))
    if unknown:
        raise ValueError(f"parameters outside the {name} search space: {unknown}")
    for key, value in params.items():
        if not space[key].contains(value):
            raise ValueError(f"{name}.{key}={value!r} is outside {space[key]}")
