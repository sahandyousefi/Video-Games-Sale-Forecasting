"""Command-line entry point.

``vgsales evaluate`` compares the configured models under expanding-window
cross-validation on the development years, all on the same folds. It is the
model selection step and never reads the holdout. Configured models whose
optional library is not installed are skipped with a note; models named with
``--models`` must be available.

``vgsales train`` fits one model on all development years, scores it once on
the holdout of the latest release years and saves the fitted estimator.

Both commands read the extract from ``data.raw_path`` unless ``--data`` or
``--synthetic`` is given, and write JSON and markdown reports to
``output.report_dir``. Scores computed on synthetic data describe the
generator, not the market; they are labelled as such and written only to an
explicit ``--report-dir``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from vgsales import __version__
from vgsales.config import DEFAULT_CONFIG_PATH, Config, load_config
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import Dataset, load_dataset, prepare_dataset
from vgsales.evaluation.cv import compare_models, comparison_table
from vgsales.evaluation.metrics import PRIMARY_METRIC, regression_metrics
from vgsales.evaluation.reports import (
    SYNTHETIC,
    paired_tests,
    write_cv_report,
    write_holdout_report,
)
from vgsales.features.prerelease import GROUP_COLUMN
from vgsales.models.registry import is_available, make_model, missing_dependency
from vgsales.models.target import with_target_transform

SUMMARY_METRICS = ("rmsle", "mae_log", "r2_log", "rmse", "mae")


def build_model(name: str, config: Config) -> BaseEstimator:
    """Build the unfitted model ``name`` with its configured parameters."""
    return make_model(name, config)


def _models_to_evaluate(
    args: argparse.Namespace, config: Config
) -> tuple[dict[str, BaseEstimator], list[str]]:
    if args.models:
        return {name: build_model(name, config) for name in args.models}, []
    names = config.models.evaluate
    skipped = [name for name in names if not is_available(name)]
    for name in skipped:
        print(f"skipped {name}: {missing_dependency(name)}")
    models = {name: build_model(name, config) for name in names if name not in skipped}
    if not models:
        raise ValueError("none of the configured models is available")
    return models, skipped


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
    models, skipped = _models_to_evaluate(args, config)
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
    model = with_target_transform(build_model(name, config), config.target.transform)
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
    artifact = {"model": model, "name": name, **metadata, "holdout_scores": scores}
    joblib.dump(artifact, output)
    print(f"\nsaved {output}")
    directory = _report_dir(args, config, source)
    if directory is not None:
        paths = write_holdout_report(name, scores, directory, metadata)
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

    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser(
        "evaluate",
        parents=[common],
        help="compare models with expanding-window cross-validation",
    )
    evaluate.add_argument("--models", nargs="+", help="models to compare")
    evaluate.set_defaults(handler=_evaluate)

    train = commands.add_parser(
        "train",
        parents=[common],
        help="fit on the development years and score the holdout",
    )
    train.add_argument("--model", help="model to fit (default: models.train)")
    train.add_argument("--output", type=Path, help="path of the saved model")
    train.set_defaults(handler=_train)
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
