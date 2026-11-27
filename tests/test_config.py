from datetime import date
from pathlib import Path

import pytest

from vgsales.config import Config, config_from_dict, load_config


def test_default_file_loads() -> None:
    cfg = load_config()
    assert cfg.target.column == "total_sales"
    assert cfg.target.transform == "log1p"
    assert cfg.split.holdout_years == 3
    assert "console" in cfg.features.categorical


def test_default_file_matches_dataclass_defaults() -> None:
    assert load_config() == Config()


def test_partial_mapping_uses_defaults() -> None:
    cfg = config_from_dict({"split": {"cv_folds": 3}})
    assert cfg.split.cv_folds == 3
    assert cfg.split.holdout_years == Config().split.holdout_years


def test_lists_are_stored_as_tuples() -> None:
    cfg = config_from_dict({"features": {"categorical": ["genre"], "numeric": []}})
    assert cfg.features.categorical == ("genre",)
    assert cfg.features.numeric == ()


def test_round_trip_through_dict() -> None:
    cfg = load_config()
    assert config_from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": 1},
        {"split": {"folds": 5}},
        {"split": {"cv_folds": 1}},
        {"split": {"holdout_years": 0}},
        {"target": {"transform": "sqrt"}},
        {"data": {"min_release_year": 2000, "max_release_year": 1990}},
        {"data": {"snapshot_date": "2024-13-01"}},
        {"features": {"categorical": ["genre"], "numeric": ["genre"]}},
        {"features": {"target_encoding_smoothing": -1.0}},
    ],
)
def test_invalid_values_raise_value_error(raw: dict) -> None:
    with pytest.raises(ValueError):
        config_from_dict(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {"seed": "42"},
        {"split": {"cv_folds": 5.0}},
        {"split": {"gap_years": True}},
        {"data": []},
    ],
)
def test_wrong_types_raise_type_error(raw: dict) -> None:
    with pytest.raises(TypeError):
        config_from_dict(raw)


def test_load_from_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text("seed: 7\nsplit:\n  holdout_years: 2\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.seed == 7
    assert cfg.split.holdout_years == 2


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == Config()


def test_unquoted_yaml_date_is_normalised(tmp_path: Path) -> None:
    path = tmp_path / "dated.yaml"
    path.write_text("data:\n  snapshot_date: 2024-06-30\n", encoding="utf-8")
    assert load_config(path).data.snapshot_date == "2024-06-30"


def test_model_section_defaults() -> None:
    models = load_config().models
    assert models.evaluate == (
        "global_median",
        "group_median",
        "ridge",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
    )
    assert models.train == "ridge"
    assert models.params_for("ridge") == {"alpha": 1.0}
    assert models.params_for("global_median") == {}


def test_model_params_are_copied() -> None:
    models = load_config().models
    models.params_for("ridge")["alpha"] = 99.0
    assert models.params_for("ridge") == {"alpha": 1.0}


def test_omitted_params_default_to_the_listed_models() -> None:
    cfg = config_from_dict({"models": {"evaluate": ["global_median", "xgboost"]}})
    assert set(cfg.models.params) == {"xgboost", "ridge"}
    assert cfg.models.params_for("xgboost")["n_estimators"] == 2000
    assert cfg.models.params_for("global_median") == {}


def test_model_section_from_mapping() -> None:
    cfg = config_from_dict(
        {"models": {"evaluate": ["ridge"], "train": "ridge", "params": {}}}
    )
    assert cfg.models.evaluate == ("ridge",)
    assert cfg.models.params == {}
    assert cfg.output.model_dir == "models"


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"models": {"evaluate": []}}, ValueError),
        ({"models": {"evaluate": ["ridge", "ridge"]}}, ValueError),
        (
            {"models": {"evaluate": ["ridge"], "params": {"xgboost": {"depth": 3}}}},
            ValueError,
        ),
        ({"models": {"params": {"ridge": 1.0}}}, TypeError),
        ({"models": {"params": [1]}}, TypeError),
        ({"models": {"train": ""}}, TypeError),
        ({"output": {"figures": "x"}}, ValueError),
    ],
)
def test_invalid_model_and_output_sections(raw: dict, error: type) -> None:
    with pytest.raises(error):
        config_from_dict(raw)


def test_series_section_defaults() -> None:
    series = load_config().series
    assert series.by == ("genre", "platform_family")
    assert series.reference in series.models
    assert series.params_for("gru")["input_length"] == 24
    assert series.params_for("ets") == {}
    plain = config_from_dict({"series": {"models": ["ets", "seasonal_naive"]}})
    assert plain.series.params == {}
    dated = config_from_dict({"series": {"start": date(2000, 1, 1)}})
    assert dated.series.start == "2000-01-01"


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"series": {"models": ["arima"]}}, ValueError),
        ({"series": {"models": ["ets"], "reference": "seasonal_naive"}}, ValueError),
        ({"series": {"by": []}}, ValueError),
        ({"series": {"by": ["na_sales"]}}, ValueError),
        ({"series": {"horizon": 0}}, ValueError),
        ({"series": {"min_history": 1}}, ValueError),
        (
            {"series": {"models": ["ets"], "reference": "ets", "params": {"gru": {}}}},
            ValueError,
        ),
        ({"series": {"params": {"gru": 3}}}, TypeError),
        ({"series": {"n_origins": 2.5}}, TypeError),
    ],
)
def test_invalid_series_section(raw: dict, error: type) -> None:
    with pytest.raises(error):
        config_from_dict(raw)


def test_tuning_section_defaults() -> None:
    cfg = load_config()
    assert cfg.tuning.n_trials == 50
    assert cfg.tuning.timeout is None
    assert cfg.tuning.metric == "rmsle"
    assert cfg.output.tuned_dir == "configs/tuned"
    budget = config_from_dict({"tuning": {"n_trials": None, "timeout": 600}})
    assert budget.tuning.n_trials is None
    assert budget.tuning.timeout == 600


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"tuning": {"n_trials": None}}, ValueError),
        ({"tuning": {"n_trials": 0}}, ValueError),
        ({"tuning": {"n_trials": 2.5}}, TypeError),
        ({"tuning": {"timeout": 0}}, ValueError),
        ({"tuning": {"timeout": "1h"}}, TypeError),
        ({"tuning": {"metric": "mape"}}, ValueError),
        ({"tuning": {"n_warmup_steps": -1}}, ValueError),
        ({"tuning": {"sampler": "random"}}, ValueError),
    ],
)
def test_invalid_tuning_section(raw: dict, error: type) -> None:
    with pytest.raises(error):
        config_from_dict(raw)


def test_interval_section_defaults() -> None:
    cfg = load_config()
    assert cfg.intervals.alpha == pytest.approx(0.1)
    assert cfg.intervals.quantile_params["validation_years"] == 2
    assert config_from_dict({"intervals": {"alpha": 0.2}}).intervals.alpha == 0.2
    custom = config_from_dict({"intervals": {"quantile_params": {"num_leaves": 15}}})
    assert custom.intervals.quantile_params == {"num_leaves": 15}


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"intervals": {"alpha": 0.0}}, ValueError),
        ({"intervals": {"alpha": 1}}, ValueError),
        ({"intervals": {"alpha": "0.1"}}, TypeError),
        ({"intervals": {"alpha": True}}, TypeError),
        ({"intervals": {"quantile_params": [1, 2]}}, TypeError),
        ({"intervals": {"quantile_params": {"alpha": 0.3}}}, ValueError),
        ({"intervals": {"quantile_params": {"quantile": 0.3}}}, ValueError),
        ({"intervals": {"level": 0.9}}, ValueError),
    ],
)
def test_invalid_interval_section(raw: dict, error: type) -> None:
    with pytest.raises(error):
        config_from_dict(raw)
