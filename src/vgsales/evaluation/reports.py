"""JSON and markdown reports of cross-validation and holdout scores.

Every report carries the metadata needed to reproduce it: the package
version, the data source, the configuration and the sizes and year ranges of
the development and holdout sets. JSON files hold all metrics at full
precision, with undefined values (for example the spread of a single fold)
written as ``null``. Markdown files hold the tables meant for reading.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vgsales.evaluation.cv import CVResult, comparison_table
from vgsales.evaluation.metrics import GREATER_IS_BETTER, PRIMARY_METRIC

CV_JSON = "cv_metrics.json"
CV_MARKDOWN = "cv_comparison.md"
SYNTHETIC = "synthetic"

_POOLED_COLUMNS = ("rmsle", "mae_log", "r2_log", "rmse", "mae")
_LABELS = {
    "rmsle": "RMSLE",
    "mae_log": "MAE log",
    "medae_log": "MedAE log",
    "r2_log": "R^2 log",
    "rmse": "RMSE",
    "mae": "MAE",
    "medae": "MedAE",
    "r2": "R^2",
}


def to_jsonable(value: Any) -> Any:
    """Convert numpy, pandas and non-finite values to strict JSON types."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path | pd.Timestamp):
        return str(value)
    return value


def markdown_table(
    frame: pd.DataFrame, float_format: str = "{:.4f}", index: bool = True
) -> str:
    """Render a frame as a GitHub-flavoured markdown table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to render.
    float_format : str, default "{:.4f}"
        Format applied to float cells; missing values are rendered as ``n/a``.
    index : bool, default True
        Include the index as the first column.

    Returns
    -------
    str
        Markdown table without a trailing newline.
    """
    table = frame.reset_index() if index else frame

    def cell(value: Any) -> str:
        if isinstance(value, float | np.floating):
            return "n/a" if np.isnan(value) else float_format.format(value)
        return str(value).replace("|", "\\|")

    header = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_lines(metadata: Mapping[str, Any]) -> list[str]:
    source = metadata.get("source", "unknown")
    lines = [f"- Source: {source}"]
    if source == SYNTHETIC:
        lines.append(
            "- The data are synthetic. Scores describe the generator and carry "
            "no information about the market."
        )
    lines.append(
        f"- Package version {metadata.get('version', 'unknown')}, "
        f"generated {metadata.get('created', _now())}"
    )
    dataset = metadata.get("dataset")
    if dataset:
        dev, test = dataset["development"], dataset["holdout"]
        lines.append(
            f"- Development: {dev['n_rows']} rows released "
            f"{dev['first_year']} to {dev['last_year']}; holdout: "
            f"{test['n_rows']} rows released {test['first_year']} to "
            f"{test['last_year']}, of which {test['n_censored']} right-censored"
        )
    transform = metadata.get("config", {}).get("target", {}).get("transform")
    if transform:
        lines.append(
            f"- Models are fitted on the {transform} scale of total sales; all "
            "metrics score back-transformed predictions in millions of units"
        )
    return lines


def _year_range(start: pd.Series, end: pd.Series) -> list[str]:
    return [str(a) if a == b else f"{a}-{b}" for a, b in zip(start, end, strict=True)]


def _fold_table(result: CVResult) -> pd.DataFrame:
    folds = result.per_fold
    return pd.DataFrame(
        {
            "fold": folds["fold"],
            "training years": _year_range(folds["train_start"], folds["train_end"]),
            "validation years": _year_range(folds["val_start"], folds["val_end"]),
            "n_train": folds["n_train"],
            "n_val": folds["n_val"],
            "n_purged": folds["n_purged"],
        }
    )


def _score_table(results: Mapping[str, CVResult]) -> pd.DataFrame:
    table = comparison_table(dict(results))
    out = pd.DataFrame(index=table.index)
    mean, std = table[f"{PRIMARY_METRIC}_mean"], table[f"{PRIMARY_METRIC}_std"]
    label = _LABELS[PRIMARY_METRIC]
    out[f"{label} mean (sd)"] = [
        f"{m:.4f} (n/a)" if np.isnan(s) else f"{m:.4f} ({s:.4f})"
        for m, s in zip(mean, std, strict=True)
    ]
    for metric in _POOLED_COLUMNS:
        column = f"{metric}_pooled"
        if column in table.columns:
            out[f"{_LABELS[metric]} pooled"] = table[column]
    return out


def cv_markdown(
    results: Mapping[str, CVResult], metadata: Mapping[str, Any] | None = None
) -> str:
    """Render the cross-validation comparison as a markdown document.

    Parameters
    ----------
    results : mapping of str to CVResult
        Results keyed by model name, all computed on the same folds.
    metadata : mapping, optional
        Keys ``source``, ``version``, ``created``, ``config`` and
        ``dataset`` as written by the command line.

    Returns
    -------
    str
        Markdown document.
    """
    if not results:
        raise ValueError("results must not be empty")
    metadata = metadata or {}
    first = next(iter(results.values()))
    better = ", ".join(_LABELS[m] for m in sorted(GREATER_IS_BETTER))
    sections = [
        "# Cross-validation comparison",
        "\n".join(_source_lines(metadata)),
        "## Folds",
        "Expanding window over release years. Training rows whose title has "
        "rows in the validation block are purged.",
        markdown_table(_fold_table(first), index=False),
        "## Scores",
        f"{_LABELS[PRIMARY_METRIC]} is given as the mean and sample standard "
        "deviation across folds. Pooled columns are computed once on the "
        "out-of-fold predictions of all folds. Models are sorted by pooled "
        f"{_LABELS[PRIMARY_METRIC]}. Lower is better except for {better}.",
        markdown_table(_score_table(results)),
    ]
    return "\n\n".join(sections) + "\n"


def cv_record(
    results: Mapping[str, CVResult], metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Collect metadata, per-fold scores and aggregates in a JSON-ready dict."""
    models = {
        name: {
            "summary": result.summary().to_dict(orient="index"),
            "per_fold": result.per_fold.to_dict(orient="records"),
        }
        for name, result in results.items()
    }
    return to_jsonable({"metadata": dict(metadata or {}), "models": models})


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _dump(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, allow_nan=False) + "\n"


def write_cv_report(
    results: Mapping[str, CVResult],
    directory: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``cv_metrics.json`` and ``cv_comparison.md`` to ``directory``.

    Returns
    -------
    tuple of Path
        Paths of the JSON and markdown files.
    """
    metadata = {"created": _now(), **(metadata or {})}
    directory = Path(directory)
    json_path = _write(directory / CV_JSON, _dump(cv_record(results, metadata)))
    md_path = _write(directory / CV_MARKDOWN, cv_markdown(results, metadata))
    return json_path, md_path


def write_holdout_report(
    name: str,
    scores: Mapping[str, Mapping[str, float]],
    directory: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``holdout_<name>.json`` and ``holdout_<name>.md`` to ``directory``.

    Parameters
    ----------
    name : str
        Model name.
    scores : mapping
        Metric dictionaries keyed by row subset, for example ``all`` and
        ``uncensored``.
    directory : str or Path
        Output directory, created if needed.
    metadata : mapping, optional
        As for :func:`cv_markdown`.

    Returns
    -------
    tuple of Path
        Paths of the JSON and markdown files.
    """
    metadata = {"created": _now(), **(metadata or {})}
    directory = Path(directory)
    record = to_jsonable({"metadata": metadata, "model": name, "scores": scores})
    table = pd.DataFrame(scores).T.rename(columns=_LABELS)
    table.index.name = "rows"
    markdown = "\n\n".join(
        [
            f"# Holdout scores: {name}",
            "\n".join(_source_lines(metadata)),
            "The model is fitted once on all development years and scored on "
            "the holdout of the latest release years. Uncensored rows exclude "
            "releases inside the censoring window, whose sales are still "
            "accumulating.",
            markdown_table(table),
        ]
    )
    json_path = _write(directory / f"holdout_{name}.json", _dump(record))
    md_path = _write(directory / f"holdout_{name}.md", markdown + "\n")
    return json_path, md_path
