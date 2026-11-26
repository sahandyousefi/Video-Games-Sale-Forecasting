r"""Hyperparameter search with Optuna on the expanding-window folds.

Let :math:`\theta` be a configuration drawn from the search space of a model
(:mod:`vgsales.tuning.spaces`) and :math:`F_1, \dots, F_K` the
expanding-window folds of the development years
(:mod:`vgsales.data.splits`), ordered from the earliest to the latest
validation block. Write :math:`m_k(\theta)` for the score, by default the
RMSLE, of the model fitted on the training rows of :math:`F_k` and evaluated
on its validation rows. The objective is the fold mean

.. math::

    L(\theta) = \frac{1}{K} \sum_{k=1}^{K} m_k(\theta),

which weights every validation year equally, as the fold mean of
:func:`vgsales.evaluation.cv.cross_validate` does. The holdout years are not
part of the development rows and never enter the search.

Configurations are proposed by the tree-structured Parzen estimator
(Bergstra et al., 2011) of Optuna (Akiba et al., 2019), seeded with
``config.seed``. After fold :math:`k` the running mean
:math:`\bar m_{1:k}(\theta)` is reported as the intermediate value of step
:math:`k`. The median stopping rule (Golovin et al., 2017) stops a trial when
its running mean at step :math:`k` is worse than the median of the running
means of earlier trials at the same step. Every trial is evaluated on the
same folds in the same order, so the comparison at step :math:`k` involves
identical validation years. The earliest folds have the shortest training
windows and are the cheapest to fit, so a poor configuration is usually
stopped before the expensive late folds. Pruning starts after
``n_startup_trials`` completed trials and never before ``n_warmup_steps``
folds have been scored. The intermediate value at the last step equals the
objective, so pruned and completed trials are compared on one scale.

Optuna is an optional dependency (``pip install "vgsales[tune]"``) and is
imported when a search runs.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from vgsales.config import Config
from vgsales.data.splits import ExpandingWindowSplit, Fold
from vgsales.evaluation.cv import check_target, fit_fold, resolve_folds
from vgsales.evaluation.metrics import (
    GREATER_IS_BETTER,
    METRIC_NAMES,
    PRIMARY_METRIC,
    regression_metrics,
)
from vgsales.models.registry import make_model
from vgsales.tuning.spaces import SearchSpace, get_space, suggest_params

FOLD_SCORES_ATTR: Final = "fold_scores"
TRIAL_COLUMNS: Final = (
    "number",
    "state",
    "value",
    "partial_value",
    "n_folds",
    "seconds",
)


def _require_optuna() -> ModuleType:
    try:
        return importlib.import_module("optuna")
    except ImportError as error:
        raise ImportError(
            "optuna is not installed; install the tuning extras with "
            "pip install 'vgsales[tune]'"
        ) from error


def direction_of(metric: str) -> str:
    """Return the Optuna direction, ``"minimize"`` or ``"maximize"``, of a metric.

    Raises
    ------
    ValueError
        If ``metric`` is unknown.
    """
    if metric not in METRIC_NAMES:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRIC_NAMES}")
    return "maximize" if metric in GREATER_IS_BETTER else "minimize"


class FoldObjective:
    """Optuna objective: the fold-mean score of one model configuration.

    Parameters
    ----------
    name : str
        Registered model name.
    config : Config
        Pipeline configuration; configured parameters of ``name`` are the
        starting point that sampled parameters override.
    X : pandas.DataFrame
        Development features.
    target : numpy.ndarray
        Observed development sales on the original scale.
    folds : list of Fold
        Folds in the order in which they are scored.
    space : mapping of str to Param
        Search space.
    fixed_params : mapping, optional
        Keyword arguments applied in every trial, for example a smaller
        ``n_estimators``. They must not overlap the search space.
    target_transform : {"log1p", "identity"}, default "log1p"
        Scale on which the models are fitted.
    metric : str, default "rmsle"
        Score of each fold, from :data:`vgsales.evaluation.metrics.METRIC_NAMES`.
    """

    def __init__(
        self,
        name: str,
        config: Config,
        X: pd.DataFrame,
        target: np.ndarray,
        folds: Sequence[Fold],
        space: SearchSpace,
        *,
        fixed_params: Mapping[str, Any] | None = None,
        target_transform: str = "log1p",
        metric: str = PRIMARY_METRIC,
    ) -> None:
        fixed = dict(fixed_params or {})
        overlap = sorted(set(fixed) & set(space))
        if overlap:
            raise ValueError(f"parameters both fixed and searched: {overlap}")
        if not folds:
            raise ValueError("folds must not be empty")
        self.name = name
        self.config = config
        self.X = X
        self.target = target
        self.folds = list(folds)
        self.space = dict(space)
        self.fixed_params = fixed
        self.target_transform = target_transform
        self.metric = metric
        self.direction = direction_of(metric)

    def __call__(self, trial: Any) -> float:
        """Score the configuration sampled by ``trial``, pruning between folds."""
        optuna = _require_optuna()
        params = suggest_params(trial, self.space)
        estimator = make_model(self.name, self.config, **self.fixed_params, **params)
        scores: list[float] = []
        for k, fold in enumerate(self.folds):
            y_pred, _ = fit_fold(
                estimator, self.X, self.target, fold, self.target_transform
            )
            y_true = self.target[fold.val_idx]
            scores.append(
                regression_metrics(y_true, y_pred, (self.metric,))[self.metric]
            )
            running = float(np.mean(scores))
            trial.report(running, step=k)
            trial.set_user_attr(FOLD_SCORES_ATTR, list(scores))
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after fold {k}")
        return running


@dataclass(frozen=True)
class TuningResult:
    """Outcome of a hyperparameter search.

    Attributes
    ----------
    name : str
        Model name.
    metric : str
        Score that was optimized.
    direction : {"minimize", "maximize"}
        Optimization direction.
    best_params : dict
        Sampled parameters of the best completed trial.
    best_value : float
        Objective of the best trial: the fold mean of ``metric``.
    best_fold_scores : tuple of float
        Per-fold scores of the best trial, earliest fold first.
    fixed_params : dict
        Parameters held fixed in every trial.
    n_folds : int
        Number of folds of a completed trial.
    seed : int
        Seed of the sampler.
    trials : pandas.DataFrame
        One row per trial with columns ``number``, ``state``, ``value`` (the
        objective, ``NaN`` unless the trial completed), ``partial_value``
        (running mean at the last scored fold), ``n_folds`` (folds scored),
        ``seconds`` and one column per sampled parameter.
    """

    name: str
    metric: str
    direction: str
    best_params: dict[str, Any]
    best_value: float
    best_fold_scores: tuple[float, ...]
    fixed_params: dict[str, Any]
    n_folds: int
    seed: int
    trials: pd.DataFrame

    def n_trials(self, state: str | None = None) -> int:
        """Number of trials, optionally only those in ``state`` (e.g. "pruned")."""
        if state is None:
            return len(self.trials)
        return int((self.trials["state"] == state).sum())


def trials_table(study: Any) -> pd.DataFrame:
    """Tabulate the trials of an Optuna study.

    Returns
    -------
    pandas.DataFrame
        See :attr:`TuningResult.trials`.
    """
    rows = []
    for trial in study.trials:
        state = trial.state.name.lower()
        steps = trial.intermediate_values
        # Optuna stores the last intermediate value as the value of a pruned
        # trial; only completed trials carry the objective.
        value = trial.value if state == "complete" else None
        rows.append(
            {
                "number": trial.number,
                "state": state,
                "value": np.nan if value is None else float(value),
                "partial_value": float(steps[max(steps)]) if steps else np.nan,
                "n_folds": len(steps),
                "seconds": (
                    trial.duration.total_seconds() if trial.duration else np.nan
                ),
                **trial.params,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=list(TRIAL_COLUMNS))
    return table


def tune(
    name: str,
    config: Config,
    X: pd.DataFrame,
    y: ArrayLike,
    cv: ExpandingWindowSplit | Sequence[Fold],
    *,
    n_trials: int | None = 50,
    timeout: float | None = None,
    groups: ArrayLike | None = None,
    space: SearchSpace | None = None,
    fixed_params: Mapping[str, Any] | None = None,
    metric: str = PRIMARY_METRIC,
    n_startup_trials: int = 5,
    n_warmup_steps: int = 1,
    seed: int | None = None,
) -> TuningResult:
    """Search the hyperparameters of a model on the expanding-window folds.

    Parameters
    ----------
    name : str
        Registered model name.
    config : Config
        Pipeline configuration. ``target.transform`` sets the modelling scale
        and ``seed`` seeds the sampler unless ``seed`` is given.
    X : pandas.DataFrame
        Development features; the holdout must not be included.
    y : array-like of float
        Observed development sales on the original scale.
    cv : ExpandingWindowSplit or sequence of Fold
        Splitter or precomputed folds. The folds are computed once and shared
        by all trials.
    n_trials : int or None, default 50
        Number of trials. None runs until ``timeout``.
    timeout : float, optional
        Time budget in seconds.
    groups : array-like, optional
        Group labels passed to the splitter for purging.
    space : mapping of str to Param, optional
        Search space. Defaults to ``get_space(name)``.
    fixed_params : mapping, optional
        Keyword arguments applied in every trial.
    metric : str, default "rmsle"
        Score of each fold.
    n_startup_trials : int, default 5
        Completed trials before the median rule may prune.
    n_warmup_steps : int, default 1
        Folds scored before a trial may be pruned.
    seed : int, optional
        Seed of the sampler. Defaults to ``config.seed``.

    Returns
    -------
    TuningResult
        Best parameters, their per-fold scores and the trial table.

    Raises
    ------
    ValueError
        If neither ``n_trials`` nor ``timeout`` is given, or no trial
        completes.
    ImportError
        If Optuna is not installed.
    """
    if n_trials is None and timeout is None:
        raise ValueError("give n_trials, timeout or both")
    if n_trials is not None and n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    optuna = _require_optuna()
    target = check_target(y, len(X))
    folds = resolve_folds(cv, X, groups)
    objective = FoldObjective(
        name,
        config,
        X,
        target,
        folds,
        get_space(name) if space is None else space,
        fixed_params=fixed_params,
        target_transform=config.target.transform,
        metric=metric,
    )
    seed = config.seed if seed is None else seed
    study = optuna.create_study(
        direction=objective.direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps
        ),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    complete = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if not complete:
        raise ValueError(f"no trial of the {name} search completed")
    best = study.best_trial
    return TuningResult(
        name=name,
        metric=metric,
        direction=objective.direction,
        best_params=dict(best.params),
        best_value=float(best.value),
        best_fold_scores=tuple(best.user_attrs[FOLD_SCORES_ATTR]),
        fixed_params=dict(objective.fixed_params),
        n_folds=len(folds),
        seed=seed,
        trials=trials_table(study),
    )
