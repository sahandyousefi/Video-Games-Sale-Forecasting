"""Experiment tracking with MLflow behind an optional-dependency guard.

MLflow is installed with ``pip install "vgsales[track]"`` and imported only
when tracking is enabled in the ``tracking`` configuration section. The
commands talk to a :class:`Tracker`; when tracking is disabled they receive
the base class, whose methods do nothing, so that the code paths are the
same with and without MLflow.

Parameters are flattened to dotted keys (``models.params.lightgbm.num_leaves``)
and converted to strings. Non-finite metrics, such as the spread of a single
fold, are not logged, because MLflow stores them inconsistently across
backends. Recent MLflow releases refuse the file-based store, so the default
backend is a local SQLite database.
"""

from __future__ import annotations

import importlib.util
import math
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np

from vgsales.config import TrackingConfig

MAX_PARAM_LENGTH: Final = 6000
_INVALID_KEY = re.compile(r"[^\w\-. /]")


def mlflow_available() -> bool:
    """Return whether MLflow can be imported."""
    return importlib.util.find_spec("mlflow") is not None


def _mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:
        raise ImportError(
            'experiment tracking needs MLflow: pip install "vgsales[track]"'
        ) from error
    return mlflow


def clean_key(key: str) -> str:
    """Replace characters that MLflow does not accept in keys with ``_``."""
    return _INVALID_KEY.sub("_", str(key))


def _text(value: Any) -> str:
    if isinstance(value, list | tuple):
        value = [_text(item) for item in value]
    text = str(value)
    if len(text) > MAX_PARAM_LENGTH:
        text = text[: MAX_PARAM_LENGTH - 3] + "..."
    return text


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested mapping into dotted keys with string values.

    Parameters
    ----------
    params : mapping
        Possibly nested parameters, such as ``Config.to_dict()``.
    prefix : str, optional
        Prefix of every key.

    Returns
    -------
    dict of str to str
        Keys cleaned by :func:`clean_key`; values longer than the MLflow
        limit of 6000 characters are truncated. Empty mappings are kept as
        ``"{}"`` so that the key remains visible.
    """
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = clean_key(f"{prefix}.{key}" if prefix else str(key))
        if isinstance(value, Mapping) and value:
            flat.update(flatten_params(value, name))
        elif isinstance(value, Mapping):
            flat[name] = "{}"
        else:
            flat[name] = _text(value)
    return flat


def finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep the finite numeric metrics, with cleaned keys and float values.

    Booleans and values that cannot be converted to float are left out.
    """
    out = {}
    for key, value in metrics.items():
        if isinstance(value, bool | np.bool_):
            continue
        if not isinstance(value, int | float):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
        if math.isfinite(value):
            out[clean_key(key)] = float(value)
    return out


class Tracker:
    """Tracker that records nothing; the interface used by the commands."""

    enabled: bool = False

    @contextmanager
    def run(
        self,
        name: str,
        tags: Mapping[str, Any] | None = None,
        nested: bool = False,
    ) -> Iterator[Tracker]:
        """Open a run for the duration of a ``with`` block."""
        yield self

    @property
    def run_id(self) -> str | None:
        """Identifier of the active run, or None."""
        return None

    def log_params(self, params: Mapping[str, Any], prefix: str = "") -> None:
        """Log possibly nested parameters."""

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Log the finite metrics, optionally at a step such as a fold index."""

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        """Set tags on the active run."""

    def log_artifact(self, path: str | Path, directory: str | None = None) -> None:
        """Copy a file into the artifacts of the active run."""


class MlflowTracker(Tracker):
    """Tracker that records runs in an MLflow experiment.

    Parameters
    ----------
    experiment : str
        Experiment name; created when it does not exist.
    tracking_uri : str, optional
        Backend store, for example ``sqlite:///mlflow.db``. When None the
        MLflow default applies, which honours ``MLFLOW_TRACKING_URI``.
    artifact_location : str, optional
        Artifact root of a new experiment; MLflow chooses one when None.

    Raises
    ------
    ImportError
        If MLflow is not installed.
    """

    enabled = True

    def __init__(
        self,
        experiment: str,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ) -> None:
        self._mlflow = _mlflow()
        if tracking_uri is not None:
            self._mlflow.set_tracking_uri(tracking_uri)
        if self._mlflow.get_experiment_by_name(experiment) is None:
            self._mlflow.create_experiment(
                experiment, artifact_location=artifact_location
            )
        self.experiment_id = self._mlflow.set_experiment(experiment).experiment_id

    @contextmanager
    def run(
        self,
        name: str,
        tags: Mapping[str, Any] | None = None,
        nested: bool = False,
    ) -> Iterator[Tracker]:
        cleaned = {clean_key(k): _text(v) for k, v in (tags or {}).items()}
        with self._mlflow.start_run(run_name=name, tags=cleaned, nested=nested):
            yield self

    @property
    def run_id(self) -> str | None:
        run = self._mlflow.active_run()
        return None if run is None else run.info.run_id

    def log_params(self, params: Mapping[str, Any], prefix: str = "") -> None:
        flat = flatten_params(params, prefix)
        if flat:
            self._mlflow.log_params(flat)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        finite = finite_metrics(metrics)
        if finite:
            self._mlflow.log_metrics(finite, step=step)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        self._mlflow.set_tags({clean_key(k): _text(v) for k, v in tags.items()})

    def log_artifact(self, path: str | Path, directory: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=directory)


def make_tracker(config: TrackingConfig) -> Tracker:
    """Build the tracker described by the ``tracking`` configuration.

    Parameters
    ----------
    config : TrackingConfig
        Tracking settings.

    Returns
    -------
    Tracker
        An :class:`MlflowTracker` when tracking is enabled, otherwise a
        :class:`Tracker` that records nothing.

    Raises
    ------
    ImportError
        If tracking is enabled and MLflow is not installed.
    """
    if not config.enabled:
        return Tracker()
    return MlflowTracker(
        config.experiment,
        tracking_uri=config.tracking_uri,
        artifact_location=config.artifact_location,
    )
