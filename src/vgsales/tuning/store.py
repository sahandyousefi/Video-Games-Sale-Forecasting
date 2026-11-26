"""Tuned parameters on disk.

A search writes one YAML file per model, ``<tuned_dir>/<name>.yaml``, with
the sampled parameters of the best trial under ``params`` and its provenance:
the objective and per-fold scores, the size and outcome of the search, the
parameters held fixed, the data source and the package version.

:func:`load_tuned` reads ``params`` back and checks every name and value
against the search space of the model before the parameters can reach an
estimator. Only ``params`` is applied. Fixed parameters, for example a
reduced ``n_estimators`` used to shorten a search, are recorded for
reference and do not replace the configured values when the tuned model is
built.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml

from vgsales.evaluation.reports import tuning_record
from vgsales.tuning.search import TuningResult
from vgsales.tuning.spaces import check_params, to_model_params

TUNED_SUFFIX: Final = ".yaml"
_HEADER: Final = (
    "# Tuned by `vgsales tune` on the expanding-window folds of the development\n"
    "# years. Only `params` is applied by --tuned; the rest records provenance.\n"
)


def tuned_path(directory: str | Path, name: str) -> Path:
    """Return the path of the tuned parameters of model ``name``."""
    return Path(directory) / f"{name}{TUNED_SUFFIX}"


def save_tuned(
    result: TuningResult,
    directory: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write the best parameters of a search to ``<directory>/<name>.yaml``.

    Returns
    -------
    Path
        Path of the written file. An existing file is replaced.
    """
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = tuning_record(result, {"created": created, **(metadata or {})})
    path = tuned_path(directory, result.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(record, sort_keys=False, default_flow_style=False)
    path.write_text(_HEADER + body, encoding="utf-8")
    return path


def read_tuned(path: str | Path) -> dict[str, Any]:
    """Read a tuned-parameter file and check its structure.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file has no ``model`` name or no ``params`` mapping.
    """
    with Path(path).open(encoding="utf-8") as handle:
        record = yaml.safe_load(handle)
    if not isinstance(record, dict) or not isinstance(record.get("model"), str):
        raise ValueError(f"{path} does not name a model")
    if not isinstance(record.get("params"), dict):
        raise ValueError(f"{path} has no params mapping")
    return record


def load_tuned(name: str, directory: str | Path) -> dict[str, Any]:
    """Load the validated tuned parameters of model ``name``.

    Parameters
    ----------
    name : str
        Model name.
    directory : str or Path
        Directory of the tuned-parameter files.

    Returns
    -------
    dict
        Keyword arguments for :func:`vgsales.models.registry.make_model`,
        with searched architecture parameters turned into estimator
        arguments by :func:`vgsales.tuning.spaces.to_model_params`.

    Raises
    ------
    FileNotFoundError
        If no file exists for ``name``.
    ValueError
        If the file belongs to another model or a parameter lies outside the
        search space of ``name``.
    """
    path = tuned_path(directory, name)
    record = read_tuned(path)
    if record["model"] != name:
        raise ValueError(f"{path} holds parameters of {record['model']!r}")
    params = dict(record["params"])
    check_params(name, params)
    return to_model_params(name, params)


def tuned_overrides(
    names: Iterable[str], directory: str | Path
) -> dict[str, dict[str, Any]]:
    """Load the tuned parameters of every model in ``names`` that has a file.

    Models without a file are left out, so their configured parameters
    apply.

    Returns
    -------
    dict of str to dict
        Validated parameters keyed by model name.
    """
    return {
        name: load_tuned(name, directory)
        for name in names
        if tuned_path(directory, name).is_file()
    }
