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
