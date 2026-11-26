"""JSON and markdown reports of cross-validation, holdout and series scores.

Every report carries the metadata needed to reproduce it: the package
version, the data source, the configuration and the sizes and year ranges of
the development and holdout sets. JSON files hold all metrics at full
precision, with undefined values (for example the spread of a single fold)
written as ``null``. Markdown files hold the tables meant for reading. When
two or more models are compared on two or more folds, the cross-validation
report adds paired Wilcoxon signed-rank tests of the per-fold primary metric
against the best model (:mod:`vgsales.evaluation.significance`). The series
report holds the rolling-origin scores of the monthly forecasters
(:mod:`vgsales.evaluation.rolling`). The tuning report holds the best
configuration of a hyperparameter search, its per-fold scores and the trial
table (:mod:`vgsales.tuning.search`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vgsales.evaluation.cv import CVResult, comparison_table
from vgsales.evaluation.metrics import GREATER_IS_BETTER, PRIMARY_METRIC
from vgsales.evaluation.rolling import RollingResult, paired_origin_tests
from vgsales.evaluation.significance import paired_comparison

if TYPE_CHECKING:
    from vgsales.tuning.search import TuningResult

CV_JSON = "cv_metrics.json"
CV_MARKDOWN = "cv_comparison.md"
SERIES_JSON = "series_metrics.json"
SERIES_MARKDOWN = "series_comparison.md"
TOP_TRIALS = 10
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
    "mase": "MASE",
    "skill": "Skill",
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
    skipped = metadata.get("skipped_models")
    if skipped:
        lines.append(
            "- Configured but not evaluated (optional library not installed): "
            + ", ".join(skipped)
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


def _fold_table(folds: pd.DataFrame) -> pd.DataFrame:
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


def paired_tests(results: Mapping[str, CVResult]) -> pd.DataFrame | None:
    """Paired tests of the primary metric against the best model.

    Returns
    -------
    pandas.DataFrame or None
        Output of :func:`vgsales.evaluation.significance.paired_comparison`,
        or None when fewer than two models or two folds are available.
    """
    if len(results) < 2:
        return None
    first = next(iter(results.values()))
    if len(first.per_fold) < 2 or PRIMARY_METRIC not in first.metrics:
        return None
    return paired_comparison(results, PRIMARY_METRIC)


def _paired_section(table: pd.DataFrame) -> list[str]:
    label = _LABELS[PRIMARY_METRIC]
    reference = table["reference"].iloc[0]
    n_folds = int(table["n_folds"].iloc[0])
    out = pd.DataFrame(index=table.index)
    out["folds better"] = [
        f"{better}/{n}"
        for better, n in zip(table["n_better"], table["n_folds"], strict=True)
    ]
    out["median difference"] = table["median_diff"]
    out["W+"] = [f"{w:g}" for w in table["statistic"]]
    out["p"] = table["p_value"]
    out["p (Holm)"] = table["p_holm"]
    text = (
        f"The {label} of each fold is paired with that of {reference}, the "
        f"model with the best pooled {label}. Folds better counts the "
        f"validation years on which a model scores better than {reference}; "
        "the median difference is taken over folds, model minus reference. "
        "p is the two-sided Wilcoxon signed-rank p-value and p (Holm) adjusts "
        f"it for the {len(table)} comparisons. With {n_folds} folds the "
        f"smallest attainable p-value is {2.0 ** (1 - n_folds):.4f}. The folds "
        "share training years, so the p-values describe how consistently the "
        "ranking holds across validation years rather than the significance "
        "of a difference in generalization error."
    )
    return ["## Paired comparison", text, markdown_table(out)]


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
        markdown_table(_fold_table(first.per_fold), index=False),
        "## Scores",
        f"{_LABELS[PRIMARY_METRIC]} is given as the mean and sample standard "
        "deviation across folds. Pooled columns are computed once on the "
        "out-of-fold predictions of all folds. Models are sorted by pooled "
        f"{_LABELS[PRIMARY_METRIC]}. Lower is better except for {better}.",
        markdown_table(_score_table(results)),
    ]
    paired = paired_tests(results)
    if paired is not None:
        sections += _paired_section(paired)
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
    record: dict[str, Any] = {"metadata": dict(metadata or {}), "models": models}
    paired = paired_tests(results)
    if paired is not None:
        record["paired_tests"] = {
            "metric": PRIMARY_METRIC,
            "test": "wilcoxon_signed_rank",
            "tests": paired.to_dict(orient="index"),
        }
    return to_jsonable(record)


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


def _series_source_lines(metadata: Mapping[str, Any]) -> list[str]:
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
    series = metadata.get("series")
    if series:
        lines.append(
            f"- Series: {series['n_segments']} segments, monthly from "
            f"{series['first_month']} to {series['last_month']} "
            f"({series['n_months']} months, {series['n_titles']} tracked titles); "
            f"the last month precedes the censoring window of the "
            f"{series['snapshot']} snapshot"
        )
    skipped = metadata.get("skipped_models")
    if skipped:
        lines.append(
            "- Configured but not evaluated (optional library not installed): "
            + ", ".join(skipped)
        )
    return lines


def _series_paired(result: RollingResult, reference: str) -> pd.DataFrame | None:
    if len(result.models) < 2 or len(result.origins) < 2:
        return None
    return paired_origin_tests(result, reference)


def series_markdown(
    result: RollingResult,
    reference: str,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Render the rolling-origin comparison as a markdown document.

    Parameters
    ----------
    result : RollingResult
        Output of :func:`vgsales.evaluation.rolling.rolling_origin_evaluation`.
    reference : str
        Model against which skill and paired tests are computed.
    metadata : mapping, optional
        Keys ``source``, ``version``, ``created``, ``series`` and
        ``skipped_models`` as written by the command line.

    Returns
    -------
    str
        Markdown document.
    """
    metadata = metadata or {}
    origins = ", ".join(origin.strftime("%Y-%m") for origin in result.origins)
    summary = result.summary(reference).sort_values("rmsle")
    summary = summary.rename(columns=_LABELS)
    by_step = result.by_step()
    by_step.index.name = "step"
    sections = [
        "# Rolling-origin comparison of series forecasters",
        "\n".join(_series_source_lines(metadata)),
        "## Design",
        f"Every forecaster is refitted on the months up to each origin ({origins}) "
        f"and forecasts the next {result.horizon} months, which are then "
        "compared with the observed monthly sales. RMSLE and MAE log are "
        "computed on log(1 + sales), MAE in millions of units. MASE scales "
        "each absolute log error by the in-sample mean absolute error of the "
        "seasonal naive forecast on the history of its origin. Skill is "
        f"1 - RMSLE / RMSLE of {reference}. Lower is better except for skill.",
        "## Scores",
        markdown_table(summary),
        "## RMSLE by forecast step",
        markdown_table(by_step),
    ]
    paired = _series_paired(result, reference)
    if paired is not None:
        out = pd.DataFrame(index=paired.index.rename("model"))
        out["origins better"] = [
            f"{better}/{n}"
            for better, n in zip(paired["n_better"], paired["n_origins"], strict=True)
        ]
        out["median difference"] = paired["median_diff"]
        out["W+"] = [f"{w:g}" for w in paired["statistic"]]
        out["p"] = paired["p_value"]
        out["p (Holm)"] = paired["p_holm"]
        sections += [
            "## Paired comparison over origins",
            f"The RMSLE of each origin is paired with that of {reference}. "
            "p is the two-sided Wilcoxon signed-rank p-value and p (Holm) "
            f"adjusts it for the {len(out)} comparisons. With "
            f"{len(result.origins)} origins the smallest attainable p-value is "
            f"{2.0 ** (1 - len(result.origins)):.4f}. Successive origins share "
            "history and their forecast months can overlap, so the test checks "
            "the consistency of the ranking across origins.",
            markdown_table(out),
        ]
    return "\n\n".join(sections) + "\n"


def series_record(
    result: RollingResult,
    reference: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect metadata and rolling-origin scores in a JSON-ready dict."""
    by_origin = result.scores(("origin",)).reset_index()
    by_origin["origin"] = by_origin["origin"].dt.strftime("%Y-%m")
    record: dict[str, Any] = {
        "metadata": dict(metadata or {}),
        "origins": [origin.strftime("%Y-%m") for origin in result.origins],
        "horizon": result.horizon,
        "reference": reference,
        "summary": result.summary(reference).to_dict(orient="index"),
        "by_step": result.scores(("step",)).reset_index().to_dict(orient="records"),
        "by_origin": by_origin.to_dict(orient="records"),
    }
    paired = _series_paired(result, reference)
    if paired is not None:
        record["paired_tests"] = {
            "metric": "rmsle",
            "test": "wilcoxon_signed_rank",
            "tests": paired.to_dict(orient="index"),
        }
    return to_jsonable(record)


def write_series_report(
    result: RollingResult,
    directory: str | Path,
    reference: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``series_metrics.json`` and ``series_comparison.md`` to ``directory``.

    Returns
    -------
    tuple of Path
        Paths of the JSON and markdown files.
    """
    metadata = {"created": _now(), **(metadata or {})}
    directory = Path(directory)
    record = series_record(result, reference, metadata)
    json_path = _write(directory / SERIES_JSON, _dump(record))
    md_path = _write(
        directory / SERIES_MARKDOWN, series_markdown(result, reference, metadata)
    )
    return json_path, md_path


def tuning_record(
    result: TuningResult, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Collect the best configuration of a search and its provenance.

    Parameters
    ----------
    result : TuningResult
        Outcome of :func:`vgsales.tuning.search.tune`.
    metadata : mapping, optional
        Provenance such as ``source`` and ``version``.

    Returns
    -------
    dict
        Keys ``model``, ``params`` (sampled parameters of the best trial),
        ``objective``, ``search`` and ``metadata``, as plain Python types.
    """
    return to_jsonable(
        {
            "model": result.name,
            "params": result.best_params,
            "objective": {
                "metric": result.metric,
                "direction": result.direction,
                "value": result.best_value,
                "fold_scores": list(result.best_fold_scores),
                "n_folds": result.n_folds,
            },
            "search": {
                "n_trials": result.n_trials(),
                "n_complete": result.n_trials("complete"),
                "n_pruned": result.n_trials("pruned"),
                "seed": result.seed,
                "fixed_params": result.fixed_params,
            },
            "metadata": dict(metadata or {}),
        }
    )


def _format_param(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def tuning_markdown(
    result: TuningResult, metadata: Mapping[str, Any] | None = None
) -> str:
    """Render a hyperparameter search as a markdown document.

    Parameters
    ----------
    result : TuningResult
        Outcome of :func:`vgsales.tuning.search.tune`.
    metadata : mapping, optional
        As for :func:`cv_markdown`.

    Returns
    -------
    str
        Markdown document.
    """
    metadata = metadata or {}
    label = _LABELS.get(result.metric, result.metric)
    best = "lowest" if result.direction == "minimize" else "highest"
    folds = _fold_table(result.folds)
    folds[label] = list(result.best_fold_scores)
    params = pd.DataFrame(
        {"value": [_format_param(value) for value in result.best_params.values()]},
        index=pd.Index(list(result.best_params), name="parameter"),
    )
    complete = result.trials[result.trials["state"] == "complete"]
    top = complete.sort_values("value", ascending=result.direction == "minimize")
    top = top.drop(columns=["state", "partial_value", "n_folds"]).head(TOP_TRIALS)
    fixed = ", ".join(f"{k}={v}" for k, v in result.fixed_params.items()) or "none"
    sections = [
        f"# Hyperparameter search: {result.name}",
        "\n".join(_source_lines(metadata)),
        f"The tree-structured Parzen estimator proposed {result.n_trials()} "
        f"configurations (sampler seed {result.seed}). Completed: "
        f"{result.n_trials('complete')}; pruned by the median rule: "
        f"{result.n_trials('pruned')}. The "
        f"objective is the mean {label} over {result.n_folds} expanding-window "
        "folds of the development years; the holdout is not used. Parameters "
        f"held fixed: {fixed}. The best fold mean is optimistically biased as an "
        "estimate of the error of the selected configuration, because the same "
        "folds selected it.",
        "## Best configuration",
        f"Fold mean {label}: {result.best_value:.4f}.",
        markdown_table(params),
        markdown_table(folds, index=False),
        f"## Completed trials ({best} {label} first, at most {TOP_TRIALS})",
        markdown_table(top, index=False),
    ]
    return "\n\n".join(sections) + "\n"


def write_tuning_report(
    result: TuningResult,
    directory: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``tuning_<name>.json`` and ``tuning_<name>.md`` to ``directory``.

    The JSON file holds the record of :func:`tuning_record` and every trial.

    Returns
    -------
    tuple of Path
        Paths of the JSON and markdown files.
    """
    metadata = {"created": _now(), **(metadata or {})}
    directory = Path(directory)
    record = tuning_record(result, metadata)
    record["trials"] = to_jsonable(result.trials.to_dict(orient="records"))
    json_path = _write(directory / f"tuning_{result.name}.json", _dump(record))
    md_path = _write(
        directory / f"tuning_{result.name}.md", tuning_markdown(result, metadata)
    )
    return json_path, md_path
