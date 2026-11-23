import importlib.util

import pytest
from sklearn.pipeline import Pipeline

from vgsales.config import config_from_dict, load_config
from vgsales.models.baselines import BASELINE_NAMES
from vgsales.models.boosting import BOOSTING_NAMES, TemporalBoostingRegressor
from vgsales.models.registry import (
    MODEL_NAMES,
    REGISTRY,
    get_spec,
    is_available,
    make_model,
    missing_dependency,
    model_params,
)


@pytest.fixture
def without_xgboost(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.util.find_spec

    def find_spec(name: str, *args: object) -> object:
        return None if name == "xgboost" else original(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_registry_covers_baselines_and_boosting_models() -> None:
    assert (*BASELINE_NAMES, *BOOSTING_NAMES) == MODEL_NAMES
    for name in BASELINE_NAMES:
        assert REGISTRY[name].family == "baseline"
        assert REGISTRY[name].requires is None
        assert is_available(name)
    for name in BOOSTING_NAMES:
        assert get_spec(name).family == "boosting"
        assert get_spec(name).requires == name
        assert get_spec(name).extra == "gbm"
    with pytest.raises(ValueError, match="unknown model"):
        get_spec("adaboost")


def test_baselines_take_their_configured_parameters() -> None:
    config = config_from_dict({"models": {"params": {"ridge": {"alpha": 3.0}}}})
    model = make_model("ridge", config)
    assert isinstance(model, Pipeline)
    assert model[-1].alpha == 3.0
    assert make_model("ridge", config, alpha=0.5)[-1].alpha == 0.5


def test_boosting_parameters_come_from_the_config_and_the_seed() -> None:
    config = load_config()
    for name in BOOSTING_NAMES:
        params = model_params(name, config)
        assert params["random_state"] == config.seed
        assert params["n_estimators"] == 2000
        assert params["validation_years"] == 2
    params = model_params("xgboost", config, random_state=7, max_depth=3)
    assert params["random_state"] == 7
    assert params["max_depth"] == 3
    assert params["subsample"] == 0.8
    assert "random_state" not in model_params("ridge", config)


@pytest.mark.parametrize("name", BOOSTING_NAMES)
def test_boosting_models_are_built_from_the_config(name: str) -> None:
    pytest.importorskip(name)
    config = load_config()
    model = make_model(name, config, n_estimators=50)
    assert isinstance(model, TemporalBoostingRegressor)
    assert model.config == config.features
    assert model.n_estimators == 50
    assert model.random_state == config.seed


@pytest.mark.usefixtures("without_xgboost")
def test_missing_library_is_reported() -> None:
    assert not is_available("xgboost")
    assert "vgsales[gbm]" in missing_dependency("xgboost")
    assert missing_dependency("ridge") is None
    with pytest.raises(ImportError, match="unavailable"):
        make_model("xgboost", load_config())
