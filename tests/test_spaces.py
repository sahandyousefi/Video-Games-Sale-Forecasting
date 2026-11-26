import numpy as np
import pytest

from vgsales.config import load_config
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.models.boosting import BOOSTING_NAMES, WRAPPER_PARAMS
from vgsales.models.registry import MODEL_NAMES, is_available, make_model
from vgsales.tuning.spaces import (
    SEARCH_SPACES,
    CategoricalParam,
    FloatParam,
    IntParam,
    check_params,
    funnel,
    get_space,
    model_keys,
    suggest_params,
    to_model_params,
)


def _bounds(space: dict, bound: str) -> dict:
    index = {"low": 0, "high": -1}[bound]
    return {
        key: p.choices[index] if isinstance(p, CategoricalParam) else getattr(p, bound)
        for key, p in space.items()
    }


def test_every_boosting_model_and_the_mlp_have_a_space() -> None:
    assert set(SEARCH_SPACES) == {*BOOSTING_NAMES, "mlp"}
    assert set(SEARCH_SPACES) <= set(MODEL_NAMES)
    for space in SEARCH_SPACES.values():
        assert "learning_rate" in space
        assert "n_estimators" not in space
        assert not {"early_stopping_rounds", "validation_years"} & set(space)


def test_configured_values_lie_inside_the_spaces() -> None:
    config = load_config()
    for name, space in SEARCH_SPACES.items():
        configured = config.models.params_for(name)
        searched = {k: v for k, v in configured.items() if k in space}
        assert searched, name
        check_params(name, searched)
    configured_hidden = tuple(config.models.params_for("mlp")["hidden"])
    assert to_model_params("mlp", {"n_layers": 2, "width": 128}) == {
        "hidden": configured_hidden
    }
    check_params("mlp", {"n_layers": 2, "width": 128})


def test_parameter_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="smaller"):
        FloatParam(1.0, 1.0)
    with pytest.raises(ValueError, match="log-scale"):
        FloatParam(0.0, 1.0, log=True)
    with pytest.raises(ValueError, match="log-scale"):
        IntParam(0, 10, log=True)
    assert IntParam(1, 5).contains(3)
    assert not IntParam(1, 5).contains(True)
    assert not IntParam(1, 5).contains(2.0)
    assert FloatParam(0.5, 1.0).contains(1)

    batch = CategoricalParam([128, 256])
    assert batch.choices == (128, 256)
    assert batch.contains(256)
    assert not batch.contains(512)
    assert not CategoricalParam((0, 1)).contains(True)
    with pytest.raises(ValueError, match="at least two"):
        CategoricalParam((128,))
    with pytest.raises(ValueError, match="distinct"):
        CategoricalParam((128, 128))
    with pytest.raises(TypeError, match="str, int or float"):
        CategoricalParam(((64, 32), (128, 64)))


def test_architecture_parameters_build_a_funnel() -> None:
    assert funnel(256, 3) == (256, 128, 64)
    assert funnel(20, 3) == (20, 10, 8)
    assert funnel(512, 1) == (512,)
    with pytest.raises(ValueError, match=">= 1"):
        funnel(64, 0)

    params = {"n_layers": 3, "width": 64, "dropout": 0.2}
    assert to_model_params("mlp", params) == {"dropout": 0.2, "hidden": (64, 32, 16)}
    assert params == {"n_layers": 3, "width": 64, "dropout": 0.2}
    assert to_model_params("mlp", {"dropout": 0.2}) == {"dropout": 0.2}
    assert to_model_params("xgboost", {"max_depth": 4}) == {"max_depth": 4}
    with pytest.raises(ValueError, match="needs all of"):
        to_model_params("mlp", {"width": 64})

    assert "hidden" in model_keys("mlp", get_space("mlp"))
    assert not {"n_layers", "width"} & model_keys("mlp", get_space("mlp"))
    assert model_keys("xgboost", get_space("xgboost")) == set(get_space("xgboost"))


def test_check_params_and_get_space() -> None:
    check_params("xgboost", {"max_depth": 4, "learning_rate": 0.05})
    with pytest.raises(ValueError, match="outside the xgboost search space"):
        check_params("xgboost", {"num_leaves": 31})
    with pytest.raises(ValueError, match="outside"):
        check_params("lightgbm", {"num_leaves": 1000})
    with pytest.raises(ValueError, match="no search space"):
        get_space("ridge")
    space = get_space("catboost")
    space.pop("depth")
    assert "depth" in SEARCH_SPACES["catboost"]


def test_suggestions_follow_the_space() -> None:
    optuna = pytest.importorskip("optuna")
    space = get_space("lightgbm")
    fixed = {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20}
    fixed |= {"subsample": 0.9, "colsample_bytree": 0.7, "reg_lambda": 1.0}
    fixed |= {"min_data_per_group": 50, "cat_smooth": 10.0}
    assert suggest_params(optuna.trial.FixedTrial(fixed), space) == fixed

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for name, space in SEARCH_SPACES.items():
        for _ in range(20):
            trial = study.ask()
            params = suggest_params(trial, space)
            check_params(name, params)
            for key, param in space.items():
                distribution = trial.distributions[key]
                if isinstance(param, CategoricalParam):
                    assert tuple(distribution.choices) == param.choices
                else:
                    assert distribution.log == param.log
            study.tell(trial, 0.0)
    rates = [t.params["learning_rate"] for t in study.trials]
    assert np.mean(np.array(rates) < 0.08) > 0.4


@pytest.mark.parametrize("name", BOOSTING_NAMES)
def test_models_fit_at_the_bounds(name: str) -> None:
    if not is_available(name):
        pytest.skip(f"{name} is not installed")
    config = load_config()
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=400), seed=2), config
    )
    X, y = dataset.development()
    for bound in ("low", "high"):
        params = _bounds(SEARCH_SPACES[name], bound)
        assert not set(params) & (set(WRAPPER_PARAMS) - {"learning_rate"})
        model = make_model(name, config, n_estimators=5, **params)
        model.fit(X, np.log1p(y))
        assert np.isfinite(model.predict(X)).all()


def test_mlp_fits_at_the_bounds() -> None:
    if not is_available("mlp"):
        pytest.skip("torch is not installed")
    config = load_config()
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=300), seed=2), config
    )
    X, y = dataset.development()
    for bound in ("low", "high"):
        params = to_model_params("mlp", _bounds(SEARCH_SPACES["mlp"], bound))
        model = make_model("mlp", config, max_epochs=2, patience=1, **params)
        model.fit(X, np.log1p(y))
        assert np.isfinite(model.predict(X)).all()
        assert model.network_ is not None
