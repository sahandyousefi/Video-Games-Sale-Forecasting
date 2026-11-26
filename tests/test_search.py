import importlib

import numpy as np
import pandas as pd
import pytest

from vgsales.config import config_from_dict
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.evaluation.cv import cross_validate, resolve_folds
from vgsales.features.prerelease import GROUP_COLUMN
from vgsales.models.registry import is_available, make_model
from vgsales.tuning import search
from vgsales.tuning.search import (
    FOLD_SCORES_ATTR,
    TRIAL_COLUMNS,
    FoldObjective,
    direction_of,
    tune,
)
from vgsales.tuning.spaces import FloatParam, check_params

RIDGE_SPACE = {"alpha": FloatParam(1e-2, 1e6, log=True)}


@pytest.fixture(scope="module")
def config():
    return config_from_dict({"seed": 3, "split": {"cv_folds": 3, "min_train_years": 5}})


@pytest.fixture(scope="module")
def data(config) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=500), seed=4), config
    )
    return dataset.development()


@pytest.fixture(scope="module")
def splitter(config) -> ExpandingWindowSplit:
    return ExpandingWindowSplit.from_config(config.split, group_column=GROUP_COLUMN)


@pytest.fixture
def optuna():
    module = pytest.importorskip("optuna")
    module.logging.set_verbosity(module.logging.WARNING)
    return module


def test_direction_follows_the_metric() -> None:
    assert direction_of("rmsle") == "minimize"
    assert direction_of("mae") == "minimize"
    assert direction_of("r2_log") == "maximize"
    with pytest.raises(ValueError, match="unknown metric"):
        direction_of("mape")


def test_objective_rejects_fixed_searched_overlap(config, data, splitter) -> None:
    X, y = data
    folds = resolve_folds(splitter, X)
    with pytest.raises(ValueError, match="both fixed and searched"):
        FoldObjective(
            "ridge", config, X, y, folds, RIDGE_SPACE, fixed_params={"alpha": 1}
        )
    with pytest.raises(ValueError, match="folds must not be empty"):
        FoldObjective("ridge", config, X, y, [], RIDGE_SPACE)


def test_objective_is_the_cross_validated_fold_mean(
    optuna, config, data, splitter
) -> None:
    X, y = data
    folds = resolve_folds(splitter, X)
    objective = FoldObjective("ridge", config, X, y, folds, RIDGE_SPACE)
    trial = optuna.trial.FixedTrial({"alpha": 10.0})
    value = objective(trial)

    expected = cross_validate(make_model("ridge", config, alpha=10.0), X, y, folds)
    assert value == pytest.approx(expected.per_fold["rmsle"].mean())
    assert trial.user_attrs[FOLD_SCORES_ATTR] == pytest.approx(
        expected.per_fold["rmsle"].tolist()
    )


def test_running_means_are_reported_and_poor_trials_pruned(
    optuna, config, data, splitter
) -> None:
    X, y = data
    folds = resolve_folds(splitter, X)
    objective = FoldObjective("ridge", config, X, y, folds, RIDGE_SPACE)
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    )
    for alpha in (1e5, 2e5, 5e5, 1e-2):
        study.enqueue_trial({"alpha": alpha})
    study.optimize(objective, n_trials=4)

    first = study.trials[0]
    scores = first.user_attrs[FOLD_SCORES_ATTR]
    running = [np.mean(scores[: k + 1]) for k in range(len(folds))]
    assert [first.intermediate_values[k] for k in range(len(folds))] == pytest.approx(
        running
    )
    assert first.value == pytest.approx(running[-1])

    last = study.trials[-1]
    assert last.state == optuna.trial.TrialState.PRUNED
    assert len(last.intermediate_values) < len(folds)


def test_tune_returns_the_best_completed_trial(optuna, config, data, splitter) -> None:
    X, y = data
    result = tune(
        "ridge",
        config,
        X,
        y,
        splitter,
        n_trials=10,
        space=RIDGE_SPACE,
        n_startup_trials=2,
    )
    trials = result.trials
    assert list(trials.columns[: len(TRIAL_COLUMNS)]) == list(TRIAL_COLUMNS)
    assert "alpha" in trials.columns
    assert result.n_trials() == 10
    assert result.n_trials("complete") + result.n_trials("pruned") == 10

    complete = trials[trials["state"] == "complete"]
    assert (complete["n_folds"] == result.n_folds).all()
    assert result.best_value == pytest.approx(complete["value"].min())
    assert trials.loc[trials["state"] == "pruned", "value"].isna().all()
    assert np.mean(result.best_fold_scores) == pytest.approx(result.best_value)
    assert set(result.best_params) == {"alpha"}
    assert result.direction == "minimize"
    assert result.seed == config.seed


def test_searches_are_reproducible(optuna, config, data, splitter) -> None:
    X, y = data
    kwargs = {"n_trials": 6, "space": RIDGE_SPACE}
    first = tune("ridge", config, X, y, splitter, **kwargs)
    second = tune("ridge", config, X, y, splitter, **kwargs)
    other = tune("ridge", config, X, y, splitter, seed=config.seed + 1, **kwargs)
    assert first.trials["alpha"].tolist() == second.trials["alpha"].tolist()
    assert first.best_value == second.best_value
    assert first.trials["alpha"].tolist() != other.trials["alpha"].tolist()


def test_maximized_metrics_pick_the_largest_value(
    optuna, config, data, splitter
) -> None:
    X, y = data
    result = tune(
        "ridge", config, X, y, splitter, n_trials=5, space=RIDGE_SPACE, metric="r2_log"
    )
    assert result.direction == "maximize"
    complete = result.trials[result.trials["state"] == "complete"]
    assert result.best_value == pytest.approx(complete["value"].max())


def test_boosting_search_uses_the_model_space(optuna, config, data, splitter) -> None:
    if not is_available("lightgbm"):
        pytest.skip("lightgbm is not installed")
    X, y = data
    result = tune(
        "lightgbm",
        config,
        X,
        y,
        splitter,
        n_trials=3,
        fixed_params={"n_estimators": 20},
    )
    check_params("lightgbm", result.best_params)
    assert result.fixed_params == {"n_estimators": 20}
    assert "num_leaves" in result.trials.columns


def test_input_checks(optuna, config, data, splitter) -> None:
    X, y = data
    with pytest.raises(ValueError, match="n_trials, timeout"):
        tune("ridge", config, X, y, splitter, n_trials=None, space=RIDGE_SPACE)
    with pytest.raises(ValueError, match="n_trials must be"):
        tune("ridge", config, X, y, splitter, n_trials=0, space=RIDGE_SPACE)
    with pytest.raises(ValueError, match="no search space"):
        tune("ridge", config, X, y, splitter, n_trials=1)
    with pytest.raises(ValueError, match="finite, non-negative"):
        tune("ridge", config, X, -y, splitter, n_trials=1, space=RIDGE_SPACE)


def test_missing_optuna_has_an_install_hint(
    monkeypatch: pytest.MonkeyPatch, config, data, splitter
) -> None:
    real = importlib.import_module

    def fake(name: str, *args, **kwargs):
        if name == "optuna":
            raise ImportError("No module named 'optuna'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(search.importlib, "import_module", fake)
    X, y = data
    with pytest.raises(ImportError, match=r"vgsales\[tune\]"):
        tune("ridge", config, X, y, splitter, n_trials=1, space=RIDGE_SPACE)
