"""Typed configuration loaded from YAML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from vgsales.data.schema import assert_no_leakage
from vgsales.evaluation.metrics import METRIC_NAMES, PRIMARY_METRIC
from vgsales.models.forecasting import FORECASTER_NAMES

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


def _boosting_params(**params: Any) -> dict[str, Any]:
    return {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "early_stopping_rounds": 100,
        "validation_years": 2,
        **params,
    }


def _default_model_params() -> dict[str, dict[str, Any]]:
    return {
        "group_median": {"min_count": 5},
        "ridge": {"alpha": 1.0},
        "xgboost": _boosting_params(
            max_depth=6, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8
        ),
        "lightgbm": _boosting_params(
            num_leaves=31, min_child_samples=20, min_data_per_group=50
        ),
        "catboost": _boosting_params(depth=6, l2_leaf_reg=3.0),
        "mlp": {
            "hidden": [128, 64],
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "max_epochs": 200,
            "patience": 20,
            "validation_years": 2,
        },
    }


@dataclass(frozen=True)
class ModelConfig:
    """Models to compare and to fit, with their hyperparameters.

    ``evaluate`` lists the models compared under cross-validation and
    ``train`` names the model fitted on all development years. ``params``
    maps a model name to keyword arguments of its estimator. When ``params``
    is omitted, the default parameters of the listed models are used. Names
    are checked against the model registry when the models are built.
    """

    evaluate: tuple[str, ...] = (
        "global_median",
        "group_median",
        "ridge",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
    )
    train: str = "ridge"
    params: dict[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluate", tuple(self.evaluate))
        if not self.evaluate:
            raise ValueError("models.evaluate must name at least one model")
        if len(set(self.evaluate)) != len(self.evaluate):
            raise ValueError(f"models.evaluate has duplicates: {self.evaluate}")
        if not isinstance(self.train, str) or not self.train:
            raise TypeError("models.train must be a model name")
        if self.params is None:
            listed = {*self.evaluate, self.train}
            defaults = _default_model_params()
            object.__setattr__(
                self, "params", {k: v for k, v in defaults.items() if k in listed}
            )
        if not isinstance(self.params, Mapping):
            raise TypeError("models.params must be a mapping")
        params = {}
        for name, values in self.params.items():
            if not isinstance(values, Mapping):
                raise TypeError(f"models.params.{name} must be a mapping")
            params[name] = dict(values)
        unused = sorted(set(params) - {*self.evaluate, self.train})
        if unused:
            raise ValueError(f"models.params given for unlisted models: {unused}")
        object.__setattr__(self, "params", params)

    def params_for(self, name: str) -> dict[str, Any]:
        """Return a copy of the keyword arguments configured for ``name``."""
        return dict((self.params or {}).get(name, {}))


def _default_series_params() -> dict[str, dict[str, Any]]:
    return {
        "gru": {
            "input_length": 24,
            "hidden_size": 32,
            "segment_dim": 4,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "max_epochs": 100,
            "patience": 15,
            "validation_months": 24,
        },
    }


@dataclass(frozen=True)
class SeriesConfig:
    """Monthly aggregate series and their rolling-origin evaluation.

    The series sum the sales of tracked titles by release month for each
    segment defined by ``by`` (see :mod:`vgsales.data.series`), from
    ``start`` to the last month before the censoring window. Segments with
    fewer than ``min_titles`` titles are dropped. ``models`` are refitted at
    ``n_origins`` origins ``origin_step`` months apart, the first with at
    least ``min_history`` months, and forecast ``horizon`` months; skill is
    reported against ``reference``. ``params`` holds keyword arguments of the
    forecasters, as for :class:`ModelConfig`.
    """

    by: tuple[str, ...] = ("genre", "platform_family")
    start: str | None = "1995-01-01"
    min_titles: int = 50
    horizon: int = 12
    n_origins: int = 5
    origin_step: int = 12
    min_history: int = 60
    season: int = 12
    models: tuple[str, ...] = ("seasonal_naive", "ses", "ets", "gru")
    reference: str = "seasonal_naive"
    params: dict[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        for name in ("by", "models"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.by:
            raise ValueError("series.by must name at least one column")
        assert_no_leakage(self.by)
        if self.start is not None:
            iso = date.fromisoformat(str(self.start)).isoformat()
            object.__setattr__(self, "start", iso)
        for name in ("min_titles", "horizon", "n_origins", "origin_step"):
            _require_int(f"series.{name}", getattr(self, name), 1)
        _require_int("series.min_history", self.min_history, 2)
        _require_int("series.season", self.season, 1)
        unknown = sorted(set(self.models) - set(FORECASTER_NAMES))
        if not self.models or unknown:
            raise ValueError(
                f"series.models must name forecasters from {FORECASTER_NAMES}, "
                f"got unknown {unknown}"
            )
        if self.reference not in self.models:
            raise ValueError("series.reference must be one of series.models")
        if self.params is None:
            defaults = _default_series_params()
            object.__setattr__(
                self, "params", {k: v for k, v in defaults.items() if k in self.models}
            )
        if not isinstance(self.params, Mapping):
            raise TypeError("series.params must be a mapping")
        params = {}
        for name, values in self.params.items():
            if not isinstance(values, Mapping):
                raise TypeError(f"series.params.{name} must be a mapping")
            params[name] = dict(values)
        unused = sorted(set(params) - set(self.models))
        if unused:
            raise ValueError(f"series.params given for unlisted models: {unused}")
        object.__setattr__(self, "params", params)

    def params_for(self, name: str) -> dict[str, Any]:
        """Return a copy of the keyword arguments configured for ``name``."""
        return dict((self.params or {}).get(name, {}))


@dataclass(frozen=True)
class TuningConfig:
    """Hyperparameter search on the expanding-window folds.

    Each search runs ``n_trials`` trials, or until ``timeout`` seconds have
    passed when ``n_trials`` is None, and minimizes or maximizes the fold
    mean of ``metric``. Trials are pruned by the median rule after
    ``n_startup_trials`` completed trials, never before ``n_warmup_steps``
    folds have been scored (:mod:`vgsales.tuning.search`).
    """

    n_trials: int | None = 50
    timeout: float | None = None
    metric: str = PRIMARY_METRIC
    n_startup_trials: int = 5
    n_warmup_steps: int = 1

    def __post_init__(self) -> None:
        if self.n_trials is None and self.timeout is None:
            raise ValueError("tuning needs n_trials, timeout or both")
        if self.n_trials is not None:
            _require_int("tuning.n_trials", self.n_trials, 1)
        if self.timeout is not None:
            timeout = self.timeout
            if isinstance(timeout, bool) or not isinstance(timeout, int | float):
                raise TypeError(f"tuning.timeout must be a number, got {timeout!r}")
            if timeout <= 0:
                raise ValueError("tuning.timeout must be positive")
        if self.metric not in METRIC_NAMES:
            raise ValueError(
                f"tuning.metric must be one of {METRIC_NAMES}, got {self.metric!r}"
            )
        _require_int("tuning.n_startup_trials", self.n_startup_trials, 0)
        _require_int("tuning.n_warmup_steps", self.n_warmup_steps, 0)


def _default_quantile_params() -> dict[str, Any]:
    return _boosting_params(num_leaves=31, min_child_samples=20, min_data_per_group=50)


@dataclass(frozen=True)
class IntervalConfig:
    """Central prediction intervals and their cross-validated evaluation.

    Intervals have nominal coverage ``1 - alpha``; ``methods`` lists the
    interval methods compared on the same folds
    (:mod:`vgsales.models.intervals`), checked when they are built.
    ``quantile_params`` holds the keyword arguments of the LightGBM quantile
    models of :class:`vgsales.models.quantile.QuantileIntervalRegressor`:
    the wrapper arguments of the boosting models and booster parameters, as
    for ``models.params.lightgbm``. The quantile level of each model follows
    from ``alpha`` and cannot be set here. Split conformal intervals are
    centred on ``point_model``, a registered model built with its
    ``models.params``. Split conformal and conformalized quantile intervals
    are calibrated on the latest ``calibration_years`` of each training
    window.
    """

    alpha: float = 0.1
    methods: tuple[str, ...] = ("quantile", "conformal", "cqr")
    quantile_params: dict[str, Any] = field(default_factory=_default_quantile_params)
    point_model: str = "lightgbm"
    calibration_years: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "methods", tuple(self.methods))
        if not self.methods:
            raise ValueError("intervals.methods must name at least one method")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError(f"intervals.methods has duplicates: {self.methods}")
        if not isinstance(self.point_model, str) or not self.point_model:
            raise TypeError("intervals.point_model must be a model name")
        _require_int("intervals.calibration_years", self.calibration_years, 1)
        alpha = self.alpha
        if isinstance(alpha, bool) or not isinstance(alpha, int | float):
            raise TypeError(f"intervals.alpha must be a number, got {alpha!r}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"intervals.alpha must lie in (0, 1), got {alpha}")
        object.__setattr__(self, "alpha", float(alpha))
        if not isinstance(self.quantile_params, Mapping):
            raise TypeError("intervals.quantile_params must be a mapping")
        reserved = sorted({"alpha", "quantile"} & set(self.quantile_params))
        if reserved:
            raise ValueError(
                f"intervals.quantile_params must not set {reserved}; the quantile "
                "levels follow from intervals.alpha"
            )
        object.__setattr__(self, "quantile_params", dict(self.quantile_params))


@dataclass(frozen=True)
class OutputConfig:
    """Directories for fitted models, generated reports and tuned parameters."""

    model_dir: str = "models"
    report_dir: str = "reports"
    tuned_dir: str = "configs/tuned"


@dataclass(frozen=True)
class Config:
    """Full pipeline configuration."""

    seed: int = 42
    data: DataConfig = DataConfig()
    target: TargetConfig = TargetConfig()
    split: SplitConfig = SplitConfig()
    features: FeatureConfig = FeatureConfig()
    models: ModelConfig = field(default_factory=ModelConfig)
    series: SeriesConfig = field(default_factory=SeriesConfig)
    tuning: TuningConfig = TuningConfig()
    intervals: IntervalConfig = field(default_factory=IntervalConfig)
    output: OutputConfig = OutputConfig()

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as plain Python types."""
        return asdict(self)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "target": TargetConfig,
    "split": SplitConfig,
    "features": FeatureConfig,
    "models": ModelConfig,
    "series": SeriesConfig,
    "tuning": TuningConfig,
    "intervals": IntervalConfig,
    "output": OutputConfig,
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
        Mapping with optional keys ``seed``, ``data``, ``target``, ``split``,
        ``features``, ``models``, ``series``, ``tuning``, ``intervals`` and
        ``output``.
        Missing keys take their default values.

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
