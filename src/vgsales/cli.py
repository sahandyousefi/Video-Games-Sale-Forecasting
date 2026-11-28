"""Command-line entry point.

``vgsales evaluate`` compares the configured models under expanding-window
cross-validation on the development years, all on the same folds. It is the
model selection step and never reads the holdout. Configured models whose
optional library is not installed are skipped with a note; models named with
``--models`` must be available.

``vgsales train`` fits one model on all development years, scores it once on
the holdout of the latest release years and saves the fitted estimator.

``vgsales series`` aggregates the tracked titles into monthly series by
segment and compares the configured series forecasters from rolling origins.
The GRU is skipped with a note when PyTorch is not installed.

``vgsales intervals`` compares the configured prediction interval methods
on the same expanding-window folds, reports their coverage, width and
interval score, and their coverage by release year and by genre, with a
reliability figure when matplotlib is installed. Like ``evaluate``, it never
reads the holdout.

``vgsales explain`` fits the configured boosting models on all development
years and writes their TreeSHAP importance by input column. For the first
model it adds local explanations of single titles and, when matplotlib is
installed, dependence plots of the most important ordered columns, next to
the importance and summary figures of every model. The rows explained are
development rows. The holdout is read only with ``--holdout``, which adds
the permutation importance of the first model on the holdout and compares
its ranking with the SHAP ranking.

``vgsales tune`` searches the hyperparameters of one model with Optuna on the
same expanding-window folds and saves the best configuration to
``output.tuned_dir``. ``evaluate`` and ``train`` apply the saved parameters
with ``--tuned``; models without a tuned file keep their configured values.
Because the folds that select the parameters are the folds that
``evaluate`` scores, the cross-validation scores of a tuned model are
optimistically biased. The holdout score of ``train --tuned`` is not.

All commands read the extract from ``data.raw_path`` unless ``--data`` or
``--synthetic`` is given, and write JSON and markdown reports to
``output.report_dir``. Scores computed on synthetic data describe the
generator, not the market; they are labelled as such and written only to an
explicit ``--report-dir``. For the same reason, parameters tuned on synthetic
data are saved only to an explicit ``--output-dir``.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from vgsales import __version__
from vgsales.config import DEFAULT_CONFIG_PATH, Config, ExplainConfig, load_config
from vgsales.data.grouping import title_key
from vgsales.data.loading import load_raw
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import RELEASE_YEAR
from vgsales.dataset import (
    Dataset,
    SeriesData,
    load_dataset,
    prepare_dataset,
    prepare_series,
)
from vgsales.evaluation.coverage import coverage_table, plot_coverage_reliability
from vgsales.evaluation.cv import compare_models, comparison_table
from vgsales.evaluation.interval_cv import compare_intervals, interval_comparison_table
from vgsales.evaluation.metrics import PRIMARY_METRIC, regression_metrics
from vgsales.evaluation.reports import (
    SYNTHETIC,
    coverage_pivot,
    paired_tests,
    write_cv_report,
    write_holdout_report,
    write_interval_report,
    write_series_report,
    write_shap_report,
    write_tuning_report,
)
from vgsales.evaluation.rolling import (
    paired_origin_tests,
    rolling_origin_evaluation,
    rolling_origins,
)
from vgsales.explain.local import (
    LocalExplanation,
    explain_row,
    ordered_columns,
    plot_local_explanation,
    plot_shap_dependence,
    spread_rows,
)
from vgsales.explain.permutation import (
    PermutationResult,
    compare_rankings,
    grouped_shap_importance,
    permutation_importance,
    rank_correlation,
)
from vgsales.explain.tree_shap import (
    ShapExplanation,
    explain_trees,
    importance_table,
    plot_shap_importance,
    plot_shap_summary,
)
from vgsales.features.prerelease import GROUP_COLUMN
from vgsales.models.forecasting import SeriesForecaster, make_forecaster
from vgsales.models.intervals import interval_dependency, make_interval_model
from vgsales.models.registry import (
    get_spec,
    is_available,
    make_model,
    missing_dependency,
)
from vgsales.models.target import with_target_transform
from vgsales.tuning.search import quiet_optuna, tune
from vgsales.tuning.spaces import get_space
from vgsales.tuning.store import save_tuned, tuned_overrides, tuned_path

SUMMARY_METRICS = ("rmsle", "mae_log", "r2_log", "rmse", "mae")
INTERVAL_SUMMARY = (
    "coverage",
    "below",
    "above",
    "width_log",
    "interval_score_log",
    "interval_score",
    "rmsle",
)
COVERAGE_FIGURE = Path("figures") / "interval_coverage.png"
FIGURE_DIR = Path("figures")
LOCAL_TERMS = 8


def build_model(
    name: str, config: Config, tuned: Mapping[str, Any] | None = None
) -> BaseEstimator:
    """Build the unfitted model ``name``.

    Parameters in ``tuned`` replace the configured ones.
    """
    return make_model(name, config, **dict(tuned or {}))


def _tuned_params(
    args: argparse.Namespace, config: Config, names: Iterable[str]
) -> dict[str, dict[str, Any]]:
    if not args.tuned:
        return {}
    names = list(names)
    directory = Path(config.output.tuned_dir)
    tuned = tuned_overrides(names, directory)
    for name in _searchable(names):
        if name in tuned:
            print(f"{name}: tuned parameters from {tuned_path(directory, name)}")
        else:
            print(f"{name}: no tuned parameters in {directory}; configured values")
    return tuned


def _searchable(names: Iterable[str]) -> list[str]:
    searchable = []
    for name in names:
        try:
            get_space(name)
        except ValueError:
            continue
        searchable.append(name)
    return searchable


def _models_to_evaluate(
    args: argparse.Namespace, config: Config
) -> tuple[dict[str, BaseEstimator], list[str], dict[str, dict[str, Any]]]:
    if args.models:
        names, skipped = list(args.models), []
    else:
        names = [n for n in config.models.evaluate if is_available(n)]
        skipped = [n for n in config.models.evaluate if n not in names]
        for name in skipped:
            print(f"skipped {name}: {missing_dependency(name)}")
        if not names:
            raise ValueError("none of the configured models is available")
    tuned = _tuned_params(args, config, names)
    models = {name: build_model(name, config, tuned.get(name)) for name in names}
    return models, skipped, tuned


def _dataset(args: argparse.Namespace, config: Config) -> tuple[Dataset, str]:
    if args.synthetic is not None:
        raw = make_synthetic(SyntheticSpec(n_titles=args.synthetic), seed=config.seed)
        return prepare_dataset(raw, config), SYNTHETIC
    path = Path(args.data or config.data.raw_path)
    return load_dataset(config, path), str(path)


def _report_dir(args: argparse.Namespace, config: Config, source: str) -> Path | None:
    if args.report_dir is not None:
        return args.report_dir
    if source == SYNTHETIC:
        return None
    return Path(config.output.report_dir)


def _metadata(config: Config, dataset: Dataset, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "version": __version__,
        "config": config.to_dict(),
        "dataset": dataset.summary(),
    }


def _print_header(dataset: Dataset, source: str) -> None:
    summary = dataset.summary()
    dev, test = summary["development"], summary["holdout"]
    print(f"vgsales {__version__}; source: {source}")
    if source == SYNTHETIC:
        print("synthetic data: scores describe the generator, not the market")
    print(
        f"development: {dev['n_rows']} rows, {dev['first_year']}-{dev['last_year']}; "
        f"holdout: {test['n_rows']} rows, {test['first_year']}-{test['last_year']} "
        f"({test['n_censored']} right-censored); purged: {summary['n_purged']}"
    )


def _evaluate(args: argparse.Namespace, config: Config) -> int:
    models, skipped, tuned = _models_to_evaluate(args, config)
    dataset, source = _dataset(args, config)
    _print_header(dataset, source)
    X, y = dataset.development()
    cv = ExpandingWindowSplit.from_config(config.split, group_column=GROUP_COLUMN)
    results = compare_models(models, X, y, cv, target_transform=config.target.transform)
    folds = next(iter(results.values())).per_fold
    print("\nfolds:")
    print(folds.loc[:, "fold":"n_purged"].to_string(index=False))
    columns = [f"{PRIMARY_METRIC}_mean", f"{PRIMARY_METRIC}_std"]
    columns += [f"{metric}_pooled" for metric in SUMMARY_METRICS]
    table = comparison_table(results).loc[:, columns]
    print("\ncross-validation scores (sales in millions of units):")
    print(table.to_string(float_format="{:.4f}".format))
    paired = paired_tests(results)
    if paired is not None:
        reference = paired["reference"].iloc[0]
        print(
            f"\npaired Wilcoxon signed-rank tests of per-fold {PRIMARY_METRIC} "
            f"against {reference}:"
        )
        columns = ["n_better", "n_folds", "median_diff", "p_value", "p_holm"]
        print(paired.loc[:, columns].to_string(float_format="{:.4f}".format))
    directory = _report_dir(args, config, source)
    if directory is not None:
        metadata = _metadata(config, dataset, source)
        if skipped:
            metadata["skipped_models"] = skipped
        if tuned:
            metadata["tuned_params"] = tuned
        paths = write_cv_report(results, directory, metadata)
        print("\nwrote " + ", ".join(str(path) for path in paths))
    return 0


def _holdout_scores(
    y_true: np.ndarray, y_pred: np.ndarray, censored: np.ndarray
) -> dict[str, dict[str, float]]:
    scores = {"all": regression_metrics(y_true, y_pred)}
    if (~censored).any():
        scores["uncensored"] = regression_metrics(y_true[~censored], y_pred[~censored])
    return scores


def _train(args: argparse.Namespace, config: Config) -> int:
    name = args.model or config.models.train
    tuned = _tuned_params(args, config, [name])
    model = with_target_transform(
        build_model(name, config, tuned.get(name)), config.target.transform
    )
    dataset, source = _dataset(args, config)
    _print_header(dataset, source)
    X_dev, y_dev = dataset.development()
    model.fit(X_dev, y_dev)
    X_test, y_test = dataset.holdout()
    y_pred = model.predict(X_test)
    scores = _holdout_scores(y_test, y_pred, dataset.censored[dataset.test_idx])
    print(f"\nholdout scores of {name} (sales in millions of units):")
    print(pd.DataFrame(scores).T.to_string(float_format="{:.4f}".format))

    output = Path(args.output or Path(config.output.model_dir) / f"{name}.joblib")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(config, dataset, source)
    if name in tuned:
        metadata["tuned_params"] = tuned[name]
    artifact = {"model": model, "name": name, **metadata, "holdout_scores": scores}
    joblib.dump(artifact, output)
    print(f"\nsaved {output}")
    directory = _report_dir(args, config, source)
    if directory is not None:
        paths = write_holdout_report(name, scores, directory, metadata)
        print("wrote " + ", ".join(str(path) for path in paths))
    return 0


def _series_data(args: argparse.Namespace, config: Config) -> tuple[SeriesData, str]:
    if args.synthetic is not None:
        raw = make_synthetic(SyntheticSpec(n_titles=args.synthetic), seed=config.seed)
        return prepare_series(raw, config), SYNTHETIC
    path = Path(args.data or config.data.raw_path)
    return prepare_series(load_raw(path), config), str(path)


def _forecasters(
    args: argparse.Namespace, config: Config
) -> tuple[dict[str, SeriesForecaster], list[str]]:
    series = config.series
    names = list(args.models or series.models)
    skipped = []
    if "gru" in names and importlib.util.find_spec("torch") is None:
        if args.models:
            raise ImportError('the GRU needs PyTorch: pip install "vgsales[dl]"')
        names.remove("gru")
        skipped.append("gru")
        print('skipped gru: PyTorch is not installed (pip install "vgsales[dl]")')
    forecasters = {}
    for name in names:
        params = series.params_for(name)
        if name == "gru":
            params.setdefault("random_state", config.seed)
        forecasters[name] = make_forecaster(name, series.horizon, **params)
    return forecasters, skipped


def _series(args: argparse.Namespace, config: Config) -> int:
    forecasters, skipped = _forecasters(args, config)
    data, source = _series_data(args, config)
    summary = data.summary()
    print(f"vgsales {__version__}; source: {source}")
    if source == SYNTHETIC:
        print("synthetic data: scores describe the generator, not the market")
    print(
        f"series: {summary['n_segments']} segments, {summary['first_month']} to "
        f"{summary['last_month']} ({summary['n_months']} months, "
        f"{summary['n_titles']} tracked titles)"
    )
    series = config.series
    origins = rolling_origins(
        data.sales.index,
        series.horizon,
        series.n_origins,
        series.origin_step,
        series.min_history,
    )
    result = rolling_origin_evaluation(
        data.sales, forecasters, origins, series.horizon, series.season
    )
    reference = series.reference if series.reference in forecasters else None
    reference = reference or next(iter(forecasters))
    print(
        f"\nrolling origins: {', '.join(o.strftime('%Y-%m') for o in origins)}; "
        f"horizon {series.horizon} months"
    )
    print(f"\nscores (skill against {reference}):")
    table = result.summary(reference).sort_values("rmsle")
    print(table.to_string(float_format="{:.4f}".format))
    if len(forecasters) > 1 and len(origins) > 1:
        paired = paired_origin_tests(result, reference)
        print(f"\npaired signed-rank tests of per-origin rmsle against {reference}:")
        columns = ["n_better", "n_origins", "median_diff", "p_value", "p_holm"]
        print(paired.loc[:, columns].to_string(float_format="{:.4f}".format))
    directory = _report_dir(args, config, source)
    if directory is not None:
        metadata = {
            "source": source,
            "version": __version__,
            "config": config.to_dict(),
            "series": summary,
        }
        if skipped:
            metadata["skipped_models"] = skipped
        paths = write_series_report(result, directory, reference, metadata)
        print("\nwrote " + ", ".join(str(path) for path in paths))
    return 0


def _interval_models(
    args: argparse.Namespace, config: Config
) -> tuple[dict[str, BaseEstimator], list[str]]:
    if args.methods:
        names, skipped = list(args.methods), []
    else:
        methods = config.intervals.methods
        names = [m for m in methods if interval_dependency(m, config) is None]
        skipped = [m for m in methods if m not in names]
        for name in skipped:
            print(f"skipped {name}: {interval_dependency(name, config)}")
        if not names:
            raise ValueError("none of the configured interval methods is available")
    return {name: make_interval_model(name, config) for name in names}, skipped


def _coverage_figure(
    by_year: pd.DataFrame, by_genre: pd.DataFrame, alpha: float, directory: Path
) -> Path | None:
    if importlib.util.find_spec("matplotlib") is None:
        print('skipped the reliability figure: pip install "vgsales[viz]"')
        return None
    path = directory / COVERAGE_FIGURE
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_coverage_reliability(by_year, by_genre, alpha, path=path)
    return path


def _intervals(args: argparse.Namespace, config: Config) -> int:
    models, skipped = _interval_models(args, config)
    dataset, source = _dataset(args, config)
    _print_header(dataset, source)
    X, y = dataset.development()
    cv = ExpandingWindowSplit.from_config(config.split, group_column=GROUP_COLUMN)
    alpha = config.intervals.alpha
    results = compare_intervals(
        models, X, y, cv, alpha=alpha, target_transform=config.target.transform
    )
    folds = next(iter(results.values())).per_fold
    print("\nfolds:")
    print(folds.loc[:, "fold":"n_purged"].to_string(index=False))
    columns = ["coverage_mean", "coverage_std"]
    columns += [f"{metric}_pooled" for metric in INTERVAL_SUMMARY]
    table = interval_comparison_table(results).loc[:, columns]
    print(
        f"\ncross-validated central {1 - alpha:.0%} intervals "
        "(sales in millions of units):"
    )
    print(table.to_string(float_format="{:.4f}".format))
    by_year = coverage_table(results, X[RELEASE_YEAR])
    by_genre = coverage_table(results, X["genre"])
    print("\ncoverage by release year:")
    print(coverage_pivot(by_year, RELEASE_YEAR).to_string(float_format="{:.3f}".format))
    directory = _report_dir(args, config, source)
    if directory is not None:
        directory = Path(directory)
        metadata = _metadata(config, dataset, source)
        if skipped:
            metadata["skipped_models"] = skipped
        figure = _coverage_figure(by_year, by_genre, alpha, directory)
        if figure is not None:
            metadata["figure"] = COVERAGE_FIGURE.as_posix()
        paths = write_interval_report(results, by_year, by_genre, directory, metadata)
        written = [*paths, figure] if figure is not None else list(paths)
        print("\nwrote " + ", ".join(str(path) for path in written))
    return 0


def _explain_models(
    args: argparse.Namespace, config: Config
) -> tuple[list[str], list[str]]:
    names = list(args.models or config.explain.models)
    for name in names:
        if get_spec(name).family != "boosting":
            raise ValueError(f"TreeSHAP explains boosting models only, got {name!r}")
    if args.models:
        for name in names:
            hint = missing_dependency(name)
            if hint is not None:
                raise ImportError(f"model {name!r} is unavailable: {hint}")
        return names, []
    available = [name for name in names if is_available(name)]
    skipped = [name for name in names if name not in available]
    for name in skipped:
        print(f"skipped {name}: {missing_dependency(name)}")
    if not available:
        raise ValueError("none of the configured boosting models is available")
    return available, skipped


def _local_rows(
    X: pd.DataFrame, explanation: ShapExplanation, settings: ExplainConfig
) -> tuple[list[Any], str]:
    if settings.titles:
        keys = title_key(pd.Series(list(settings.titles), dtype=object))
        present = set(X[GROUP_COLUMN].dropna())
        for title, key in zip(settings.titles, keys, strict=True):
            if key not in present:
                print(f"title {title!r} is not among the development rows")
        rows = list(X.index[X[GROUP_COLUMN].isin(set(keys.dropna()))])
        return rows, "the development rows of the titles in explain.titles"
    predictions = pd.Series(explanation.predictions(), index=explanation.values.index)
    rows = spread_rows(predictions, settings.n_local)
    return rows, (
        f"{len(rows)} explained development rows at evenly spaced ranks of the "
        "prediction, from highest to lowest"
    )


def _local_explanations(
    model: BaseEstimator,
    explanation: ShapExplanation,
    X: pd.DataFrame,
    config: Config,
) -> tuple[list[LocalExplanation], str]:
    rows, selection = _local_rows(X, explanation, config.explain)
    if not rows:
        return [], selection
    values = explain_trees(model, X.loc[rows], max_rows=None)
    local = [
        explain_row(values, X, row, top=LOCAL_TERMS, transform=config.target.transform)
        for row in rows
    ]
    return local, selection


def _slug(name: str) -> str:
    return re.sub(r"[^\w-]+", "_", name).strip("_")


def _shap_figures(
    explanations: Mapping[str, ShapExplanation],
    local: Sequence[LocalExplanation],
    settings: ExplainConfig,
    directory: Path,
) -> dict[str, Any]:
    if importlib.util.find_spec("matplotlib") is None:
        print('skipped the SHAP figures: pip install "vgsales[explain]"')
        return {}
    (directory / FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    figures = {}
    for name, explanation in explanations.items():
        importance = FIGURE_DIR / f"shap_importance_{name}.png"
        summary = FIGURE_DIR / f"shap_summary_{name}.png"
        plot_shap_importance(
            explanation,
            top=settings.top,
            title=f"Global SHAP importance: {name}",
            path=directory / importance,
        )
        plot_shap_summary(
            explanation,
            top=settings.top,
            title=f"SHAP values by row: {name}",
            path=directory / summary,
        )
        figures[name] = {
            "importance": importance.as_posix(),
            "summary": summary.as_posix(),
        }
    out: dict[str, Any] = {"figures": figures}
    primary, explanation = next(iter(explanations.items()))
    dependence = {}
    for column in ordered_columns(explanation, settings.dependence):
        path = FIGURE_DIR / f"shap_dependence_{primary}_{_slug(column)}.png"
        colour = RELEASE_YEAR
        if column == RELEASE_YEAR or RELEASE_YEAR not in explanation.data.columns:
            colour = None
        plot_shap_dependence(
            explanation,
            column,
            color_by=colour,
            random_state=0,
            title=f"Dependence on {column}: {primary}",
            path=directory / path,
        )
        dependence[column] = path.as_posix()
    if dependence:
        out["dependence_figures"] = dependence
    local_paths = []
    for position, item in enumerate(local, start=1):
        path = FIGURE_DIR / f"shap_local_{primary}_{position}.png"
        plot_local_explanation(item, path=directory / path)
        local_paths.append(path.as_posix())
    if local_paths:
        out["local_figures"] = local_paths
    return out


def _figure_paths(figures: Mapping[str, Any]) -> list[str]:
    paths = [p for item in figures.get("figures", {}).values() for p in item.values()]
    paths += list(figures.get("dependence_figures", {}).values())
    paths += list(figures.get("local_figures", []))
    return paths


def _print_local(local: Sequence[LocalExplanation], name: str, selection: str) -> None:
    print(f"\nlocal explanations of {name} for {selection}:")
    for item in local:
        print(f"\n{item.label}: predicted {item.predicted_sales():.4f} million units")
        print(item.contributions.to_string(float_format="{:+.4f}".format))


def _holdout_permutation(
    model: BaseEstimator,
    explanation: ShapExplanation,
    dataset: Dataset,
    name: str,
    config: Config,
) -> PermutationResult:
    X_test, y_test = dataset.holdout()
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=config.explain.n_repeats,
        random_state=config.seed,
    )
    shap = grouped_shap_importance(explanation, result.groups)
    table = compare_rankings(result.importance()["importance"], shap)
    print(
        f"\npermutation importance of {name} on {len(X_test)} holdout rows "
        f"({result.metric} {result.baseline:.4f} before permutation, "
        f"{config.explain.n_repeats} repeats) and grouped SHAP importance:"
    )
    shown = table.copy()
    for column in ("permutation_rank", "shap_rank"):
        shown[column] = shown[column].map("{:g}".format)
    print(shown.to_string(float_format="{:.4f}".format))
    print(f"Spearman rank correlation: {rank_correlation(table):.3f}")
    return result


def _explain(args: argparse.Namespace, config: Config) -> int:
    names, skipped = _explain_models(args, config)
    if importlib.util.find_spec("shap") is None:
        raise ImportError(
            'SHAP values need the shap package: pip install "vgsales[explain]"'
        )
    tuned = _tuned_params(args, config, names)
    dataset, source = _dataset(args, config)
    _print_header(dataset, source)
    X, y = dataset.development()
    settings = config.explain
    models, explanations = {}, {}
    for name in names:
        model = with_target_transform(
            build_model(name, config, tuned.get(name)), config.target.transform
        )
        models[name] = model.fit(X, y)
        explanations[name] = explain_trees(
            model, X, max_rows=settings.max_rows, random_state=config.seed
        )
    primary = names[0]
    local, selection = _local_explanations(
        models[primary], explanations[primary], X, config
    )
    n_rows = len(explanations[primary].values)
    print(
        f"\nmean |SHAP value| by input column on {n_rows} of {len(X)} development "
        "rows (log(1 + sales) scale):"
    )
    print(importance_table(explanations).to_string(float_format="{:.4f}".format))
    if local:
        _print_local(local, primary, selection)
    permutation = None
    if args.holdout:
        permutation = _holdout_permutation(
            models[primary], explanations[primary], dataset, primary, config
        )
    directory = _report_dir(args, config, source)
    if directory is not None:
        directory = Path(directory)
        metadata = _metadata(config, dataset, source)
        metadata["explained_rows"] = {
            "n_rows": n_rows,
            "n_available": len(X),
            "seed": config.seed,
        }
        if skipped:
            metadata["skipped_models"] = skipped
        if tuned:
            metadata["tuned_params"] = tuned
        metadata["local_model"] = primary
        metadata["local_selection"] = selection
        figures = _shap_figures(explanations, local, settings, directory)
        metadata.update(figures)
        paths = write_shap_report(
            explanations,
            directory,
            metadata,
            top=settings.top,
            local=local,
            permutation=permutation,
        )
        written = [str(path) for path in paths]
        written += [str(directory / path) for path in _figure_paths(figures)]
        print("\nwrote " + ", ".join(written))
    return 0


def _tuned_dir(args: argparse.Namespace, config: Config, source: str) -> Path | None:
    if args.output_dir is not None:
        return args.output_dir
    if source == SYNTHETIC:
        return None
    return Path(config.output.tuned_dir)


def _tune(args: argparse.Namespace, config: Config) -> int:
    name = args.model
    get_space(name)
    hint = missing_dependency(name)
    if hint is not None:
        raise ImportError(f"model {name!r} is unavailable: {hint}")
    if not args.verbose:
        quiet_optuna()
    settings = config.tuning
    n_trials = settings.n_trials if args.n_trials is None else args.n_trials
    timeout = settings.timeout if args.timeout is None else args.timeout
    dataset, source = _dataset(args, config)
    _print_header(dataset, source)
    X, y = dataset.development()
    cv = ExpandingWindowSplit.from_config(config.split, group_column=GROUP_COLUMN)
    result = tune(
        name,
        config,
        X,
        y,
        cv,
        n_trials=n_trials,
        timeout=timeout,
        metric=settings.metric,
        n_startup_trials=settings.n_startup_trials,
        n_warmup_steps=settings.n_warmup_steps,
    )
    metric = result.metric
    print(
        f"\nsearch of {name}: {result.n_trials()} trials "
        f"({result.n_trials('complete')} completed, "
        f"{result.n_trials('pruned')} pruned), sampler seed {result.seed}"
    )
    print(f"best fold mean {metric}: {result.best_value:.4f}")
    print("best parameters:")
    for key, value in result.best_params.items():
        shown = f"{value:.6g}" if isinstance(value, float) else value
        print(f"  {key}: {shown}")
    columns = ["fold", "val_start", "val_end", "n_train", "n_val"]
    folds = result.folds.loc[:, columns].copy()
    folds[metric] = result.best_fold_scores
    print(f"\nper-fold {metric} of the best configuration:")
    print(folds.to_string(index=False, float_format="{:.4f}".format))

    directory = _tuned_dir(args, config, source)
    if directory is not None:
        provenance = {
            "source": source,
            "version": __version__,
            "dataset": dataset.summary(),
        }
        print(f"\nsaved {save_tuned(result, directory, provenance)}")
    report_dir = _report_dir(args, config, source)
    if report_dir is not None:
        metadata = _metadata(config, dataset, source)
        paths = write_tuning_report(result, report_dir, metadata)
        print("wrote " + ", ".join(str(path) for path in paths))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vgsales",
        description="Pre-release forecasting of video game unit sales.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration (default: configs/default.yaml)",
    )
    source = common.add_mutually_exclusive_group()
    source.add_argument(
        "--data", type=Path, help="CSV extract (default: data.raw_path)"
    )
    source.add_argument(
        "--synthetic",
        type=int,
        metavar="N_TITLES",
        help="use the synthetic generator with this many titles",
    )
    common.add_argument(
        "--report-dir",
        type=Path,
        help="directory for JSON and markdown reports (default: output.report_dir; "
        "synthetic runs write reports only when this is set)",
    )
    common.add_argument("-v", "--verbose", action="store_true", help="log progress")

    tuned_help = "apply tuned parameters from output.tuned_dir where they exist"
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser(
        "evaluate",
        parents=[common],
        help="compare models with expanding-window cross-validation",
    )
    evaluate.add_argument("--models", nargs="+", help="models to compare")
    evaluate.add_argument("--tuned", action="store_true", help=tuned_help)
    evaluate.set_defaults(handler=_evaluate)

    train = commands.add_parser(
        "train",
        parents=[common],
        help="fit on the development years and score the holdout",
    )
    train.add_argument("--model", help="model to fit (default: models.train)")
    train.add_argument("--output", type=Path, help="path of the saved model")
    train.add_argument("--tuned", action="store_true", help=tuned_help)
    train.set_defaults(handler=_train)

    series = commands.add_parser(
        "series",
        parents=[common],
        help="compare monthly series forecasters from rolling origins",
    )
    series.add_argument(
        "--models", nargs="+", help="forecasters to compare (default: series.models)"
    )
    series.set_defaults(handler=_series)

    intervals = commands.add_parser(
        "intervals",
        parents=[common],
        help="compare prediction interval methods with expanding-window "
        "cross-validation",
    )
    intervals.add_argument(
        "--methods", nargs="+", help="interval methods (default: intervals.methods)"
    )
    intervals.set_defaults(handler=_intervals)

    explain = commands.add_parser(
        "explain",
        parents=[common],
        help="fit the boosting models on the development years and write "
        "their TreeSHAP importance",
    )
    explain.add_argument(
        "--models", nargs="+", help="boosting models (default: explain.models)"
    )
    explain.add_argument("--tuned", action="store_true", help=tuned_help)
    explain.add_argument(
        "--holdout",
        action="store_true",
        help="also compute permutation importance on the holdout",
    )
    explain.set_defaults(handler=_explain)

    search = commands.add_parser(
        "tune",
        parents=[common],
        help="search hyperparameters with Optuna on the cross-validation folds",
    )
    search.add_argument("--model", required=True, help="model to tune")
    search.add_argument(
        "--n-trials", type=int, help="number of trials (default: tuning.n_trials)"
    )
    search.add_argument(
        "--timeout", type=float, help="time budget in seconds (default: tuning.timeout)"
    )
    search.add_argument(
        "--output-dir",
        type=Path,
        help="directory of the tuned parameters (default: output.tuned_dir; "
        "synthetic runs save parameters only when this is set)",
    )
    search.set_defaults(handler=_tune)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        return args.handler(args, config)
    except (FileNotFoundError, ImportError, ValueError, TypeError) as error:
        if args.verbose:
            raise
        print(f"vgsales: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
