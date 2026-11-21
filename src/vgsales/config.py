"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from vgsales.data.schema import assert_no_leakage

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

TARGET_TRANSFORMS = ("log1p", "identity")


def _require_int(name: str, value: Any, minimum: int | None = None) -> None:
    # bool is a subclass of int and is rejected explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


@dataclass(frozen=True)
class DataConfig:
    """Location and scope of the raw data."""

    raw_path: str = "data/raw/vgchartz-2024.csv"
    snapshot_date: str | None = None
    min_release_year: int = 1980
    max_release_year: int | None = None
    censoring_window_years: int = 2

    def __post_init__(self) -> None:
        _require_int("data.min_release_year", self.min_release_year)
        _require_int("data.censoring_window_years", self.censoring_window_years, 0)
        if self.max_release_year is not None:
            _require_int("data.max_release_year", self.max_release_year)
            if self.max_release_year < self.min_release_year:
                raise ValueError(
                    "data.max_release_year must not precede data.min_release_year"
                )
        if self.snapshot_date is not None:
            # YAML parses unquoted ISO dates into datetime.date objects.
            iso = date.fromisoformat(str(self.snapshot_date)).isoformat()
            object.__setattr__(self, "snapshot_date", iso)


@dataclass(frozen=True)
class TargetConfig:
    """Target column and the variance-stabilising transform applied to it."""

    column: str = "total_sales"
    transform: str = "log1p"

    def __post_init__(self) -> None:
        if self.transform not in TARGET_TRANSFORMS:
            raise ValueError(
                f"target.transform must be one of {TARGET_TRANSFORMS}, "
                f"got {self.transform!r}"
            )


@dataclass(frozen=True)
class SplitConfig:
    """Release-year holdout and expanding-window cross-validation settings."""

    holdout_years: int = 3
    cv_folds: int = 5
    min_train_years: int = 10
    gap_years: int = 0

    def __post_init__(self) -> None:
        _require_int("split.holdout_years", self.holdout_years, 1)
        _require_int("split.cv_folds", self.cv_folds, 2)
        _require_int("split.min_train_years", self.min_train_years, 1)
        _require_int("split.gap_years", self.gap_years, 0)


@dataclass(frozen=True)
class FeatureConfig:
    """Input columns and encoder hyperparameters.

    ``categorical`` columns are one-hot encoded with categories rarer than
    ``min_category_count`` pooled. ``high_cardinality`` columns are encoded by
    their training frequency and by the temporal target mean. ``numeric``
    columns are imputed and passed through.
    """

    categorical: tuple[str, ...] = ("console", "genre")
    high_cardinality: tuple[str, ...] = ("publisher", "developer")
    numeric: tuple[str, ...] = (
        "release_year",
        "release_month",
        "release_quarter",
        "holiday_season",
        "platform_age_years",
        "n_platforms",
        "is_port",
        "critic_score",
        "critic_missing",
    )
    min_category_count: int = 5
    target_encoding_smoothing: float = 20.0

    def __post_init__(self) -> None:
        groups = ("categorical", "high_cardinality", "numeric")
        for name in groups:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for i, first in enumerate(groups):
            for second in groups[i + 1 :]:
                overlap = set(getattr(self, first)) & set(getattr(self, second))
                if overlap:
                    raise ValueError(
                        f"columns listed as {first} and {second}: {sorted(overlap)}"
                    )
        assert_no_leakage(self.categorical + self.high_cardinality + self.numeric)
        _require_int("features.min_category_count", self.min_category_count, 1)
        if float(self.target_encoding_smoothing) < 0:
            raise ValueError("features.target_encoding_smoothing must be >= 0")


@dataclass(frozen=True)
class Config:
    """Full pipeline configuration."""

    seed: int = 42
    data: DataConfig = DataConfig()
    target: TargetConfig = TargetConfig()
    split: SplitConfig = SplitConfig()
    features: FeatureConfig = FeatureConfig()

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as plain Python types."""
        return asdict(self)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "target": TargetConfig,
    "split": SplitConfig,
    "features": FeatureConfig,
}


def _build_section(cls: type, values: Any, name: str) -> Any:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise TypeError(f"section '{name}' must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown keys in section '{name}': {sorted(unknown)}")
    return cls(**values)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build a validated configuration from a nested mapping.

    Parameters
    ----------
    raw : dict
        Mapping with optional keys ``seed``, ``data``, ``target``, ``split`` and
        ``features``. Missing keys take their default values.

    Returns
    -------
    Config
        Immutable configuration object.

    Raises
    ------
    ValueError
        If a key is unknown or a value is out of range.
    TypeError
        If a value has the wrong type.
    """
    unknown = set(raw) - {"seed", *_SECTIONS}
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    seed = raw.get("seed", 42)
    _require_int("seed", seed, 0)
    sections = {
        name: _build_section(cls, raw.get(name), name)
        for name, cls in _SECTIONS.items()
    }
    return Config(seed=seed, **sections)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the YAML file. Defaults to ``configs/default.yaml``.

    Returns
    -------
    Config
        Validated configuration.
    """
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)
