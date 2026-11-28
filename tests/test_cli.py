import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from vgsales import __version__
from vgsales.cli import build_model, main
from vgsales.config import DEFAULT_CONFIG_PATH, config_from_dict, load_config
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.models.registry import is_available
from vgsales.tuning.store import load_tuned, read_tuned

ROOT = DEFAULT_CONFIG_PATH.parents[1]
N_TITLES = "900"


@pytest.fixture
def small_config(tmp_path: Path) -> Path:
    path = tmp_path / "small.yaml"
    path.write_text(
        "seed: 3\nsplit:\n  cv_folds: 3\n  min_train_years: 5\n"
        f"output:\n  model_dir: {tmp_path / 'models'}\n"
        f"  report_dir: {tmp_path / 'reports'}\n",
        encoding="utf-8",
    )
    return path


def test_evaluate_on_synthetic_data(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evaluate", "--synthetic", N_TITLES]) == 0
    out = capsys.readouterr().out
    assert "source: synthetic" in out
    assert "scores describe the generator" in out
    for name in load_config().models.evaluate:
        assert name in out
    assert "rmsle_pooled" in out
    assert "paired Wilcoxon signed-rank tests of per-fold rmsle" in out


def test_evaluate_model_subset_and_custom_config(
    small_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["evaluate", "--config", str(small_config), "--synthetic", N_TITLES]
    assert main([*argv, "--models", "ridge"]) == 0
    out = capsys.readouterr().out
    assert "ridge" in out
    assert "group_median" not in out
    folds = out.split("folds:")[1].split("cross-validation scores")[0]
    assert len(folds.strip().splitlines()) == 1 + 3


def test_train_saves_a_fitted_model(
    small_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["train", "--config", str(small_config), "--synthetic", N_TITLES]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "holdout scores of ridge" in out
    saved = small_config.parent / "models" / "ridge.joblib"
    assert saved.is_file()

    artifact = joblib.load(saved)
    assert artifact["name"] == "ridge"
    assert artifact["version"] == __version__
    assert artifact["source"] == "synthetic"
    assert set(artifact["holdout_scores"]) <= {"all", "uncensored"}

    config = load_config(small_config)
    raw = make_synthetic(SyntheticSpec(n_titles=int(N_TITLES)), seed=config.seed)
    X_test, _ = prepare_dataset(raw, config).holdout()
    pred = artifact["model"].predict(X_test)
    assert pred.shape == (len(X_test),)
    assert np.isfinite(pred).all()


def test_train_from_csv_with_explicit_model_and_output(
    small_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = tmp_path / "extract.csv"
    make_synthetic(SyntheticSpec(n_titles=int(N_TITLES)), seed=1).to_csv(
        csv, index=False
    )
    output = tmp_path / "out" / "median.joblib"
    argv = ["train", "--config", str(small_config), "--data", str(csv)]
    assert main([*argv, "--model", "group_median", "--output", str(output)]) == 0
    assert f"source: {csv}" in capsys.readouterr().out
    assert joblib.load(output)["name"] == "group_median"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["evaluate", "--data", "missing.csv"], "not found"),
        (["evaluate", "--synthetic", N_TITLES, "--models", "xgb"], "unknown"),
        (["train", "--synthetic", "0"], "positive"),
    ],
)
def test_errors_are_reported_without_traceback(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 1
    assert message in capsys.readouterr().err


def test_verbose_mode_raises_errors() -> None:
    with pytest.raises(FileNotFoundError):
        main(["evaluate", "--data", "missing.csv", "--verbose"])


def test_data_sources_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        main(["evaluate", "--data", "a.csv", "--synthetic", "10"])


def test_build_model_uses_configured_parameters() -> None:
    config = config_from_dict({"models": {"params": {"ridge": {"alpha": 7.5}}}})
    assert build_model("ridge", config)[-1].alpha == 7.5


def test_module_entry_point() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-m", "vgsales", "--version"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert completed.stdout.strip() == __version__


def test_reports_are_written_to_an_explicit_directory(
    small_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = tmp_path / "reports"
    common = ["--config", str(small_config), "--synthetic", N_TITLES]
    assert main(["evaluate", *common, "--report-dir", str(reports)]) == 0
    assert main(["train", *common, "--report-dir", str(reports)]) == 0
    names = sorted(path.name for path in reports.iterdir())
    assert names == [
        "cv_comparison.md",
        "cv_metrics.json",
        "holdout_ridge.json",
        "holdout_ridge.md",
    ]
    assert "wrote" in capsys.readouterr().out


def test_synthetic_runs_skip_the_configured_report_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = tmp_path / "reports"
    config = tmp_path / "config.yaml"
    config.write_text(
        "split:\n  cv_folds: 3\n  min_train_years: 5\n"
        f"output:\n  report_dir: {reports}\n",
        encoding="utf-8",
    )
    assert main(["evaluate", "--config", str(config), "--synthetic", N_TITLES]) == 0
    assert not reports.exists()
    assert "wrote" not in capsys.readouterr().out


def test_csv_runs_write_to_the_configured_report_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports = tmp_path / "reports"
    config = tmp_path / "config.yaml"
    config.write_text(
        "split:\n  cv_folds: 3\n  min_train_years: 5\n"
        f"output:\n  report_dir: {reports}\n",
        encoding="utf-8",
    )
    csv = tmp_path / "extract.csv"
    make_synthetic(SyntheticSpec(n_titles=int(N_TITLES)), seed=2).to_csv(
        csv, index=False
    )
    argv = ["evaluate", "--config", str(config), "--data", str(csv)]
    assert main([*argv, "--models", "global_median"]) == 0
    assert (reports / "cv_comparison.md").is_file()


def test_unavailable_configured_models_are_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = importlib.util.find_spec

    def find_spec(name: str, *args: object) -> object:
        return None if name == "xgboost" else original(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    config = tmp_path / "config.yaml"
    config.write_text(
        "split:\n  cv_folds: 3\n  min_train_years: 5\n"
        "models:\n  evaluate: [global_median, xgboost]\n",
        encoding="utf-8",
    )
    reports = tmp_path / "reports"
    common = ["evaluate", "--config", str(config), "--synthetic", N_TITLES]
    assert main([*common, "--report-dir", str(reports)]) == 0
    out = capsys.readouterr().out
    assert "skipped xgboost: xgboost is not installed" in out
    record = json.loads((reports / "cv_metrics.json").read_text(encoding="utf-8"))
    assert list(record["models"]) == ["global_median"]
    assert record["metadata"]["skipped_models"] == ["xgboost"]
    assert "not evaluated" in (reports / "cv_comparison.md").read_text("utf-8")

    assert main([*common, "--models", "xgboost"]) == 1
    assert "unavailable" in capsys.readouterr().err


@pytest.fixture
def series_config(tmp_path: Path) -> Path:
    path = tmp_path / "series.yaml"
    path.write_text(
        "seed: 3\nseries:\n  by: [genre]\n  min_titles: 5\n  horizon: 6\n"
        "  n_origins: 3\n  min_history: 60\n"
        "  models: [seasonal_naive, ses, ets, gru]\n  params:\n    gru:\n"
        "      input_length: 12\n      hidden_size: 4\n      max_epochs: 2\n"
        "      validation_months: 6\n",
        encoding="utf-8",
    )
    return path


def test_series_compares_forecasters_and_writes_reports(
    series_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = importlib.util.find_spec

    def find_spec(name: str, *args: object) -> object:
        return None if name == "torch" else original(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    reports = tmp_path / "reports"
    argv = ["series", "--config", str(series_config), "--synthetic", "3000"]
    assert main([*argv, "--report-dir", str(reports)]) == 0
    out = capsys.readouterr().out
    assert "skipped gru: PyTorch is not installed" in out
    assert "skill against seasonal_naive" in out
    assert "paired signed-rank tests of per-origin rmsle" in out
    record = json.loads((reports / "series_metrics.json").read_text("utf-8"))
    assert set(record["summary"]) == {"seasonal_naive", "ses", "ets"}
    assert len(record["origins"]) == 3
    assert record["metadata"]["skipped_models"] == ["gru"]
    markdown = (reports / "series_comparison.md").read_text("utf-8")
    assert "The data are synthetic" in markdown
    assert "## RMSLE by forecast step" in markdown

    assert main([*argv, "--models", "gru"]) == 1


def test_series_runs_the_gru(
    series_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    argv = ["series", "--config", str(series_config), "--synthetic", "3000"]
    assert main([*argv, "--models", "seasonal_naive", "gru"]) == 0
    out = capsys.readouterr().out
    assert "gru" in out
    assert "wrote" not in out


@pytest.fixture
def tune_config(tmp_path: Path) -> Path:
    pytest.importorskip("optuna")
    if not is_available("lightgbm"):
        pytest.skip("lightgbm is not installed")
    path = tmp_path / "tune.yaml"
    path.write_text(
        "seed: 3\nsplit:\n  cv_folds: 3\n  min_train_years: 5\n"
        "models:\n  evaluate: [ridge, lightgbm]\n  train: lightgbm\n"
        "  params:\n    lightgbm:\n      n_estimators: 30\n"
        "      early_stopping_rounds: 10\n"
        "tuning:\n  n_trials: 4\n  n_startup_trials: 2\n"
        f"output:\n  tuned_dir: {tmp_path / 'tuned'}\n"
        f"  model_dir: {tmp_path / 'models'}\n",
        encoding="utf-8",
    )
    return path


def test_tuned_parameters_are_saved_and_applied(
    tune_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tuned, reports = tmp_path / "tuned", tmp_path / "reports"
    common = ["--config", str(tune_config), "--synthetic", N_TITLES]
    argv = ["tune", *common, "--model", "lightgbm", "--output-dir", str(tuned)]
    assert main([*argv, "--report-dir", str(reports)]) == 0
    out = capsys.readouterr().out
    assert "search of lightgbm: 4 trials" in out
    assert "per-fold rmsle of the best configuration" in out
    assert sorted(path.name for path in reports.iterdir()) == [
        "tuning_lightgbm.json",
        "tuning_lightgbm.md",
    ]
    best = load_tuned("lightgbm", tuned)
    fold_scores = read_tuned(tuned / "lightgbm.yaml")["objective"]["fold_scores"]

    assert main(["evaluate", *common, "--tuned", "--report-dir", str(reports)]) == 0
    out = capsys.readouterr().out
    assert f"lightgbm: tuned parameters from {tuned / 'lightgbm.yaml'}" in out
    assert "ridge:" not in out
    record = json.loads((reports / "cv_metrics.json").read_text(encoding="utf-8"))
    assert record["metadata"]["tuned_params"] == {"lightgbm": best}
    per_fold = record["models"]["lightgbm"]["per_fold"]
    assert [fold["rmsle"] for fold in per_fold] == pytest.approx(fold_scores)

    assert main(["train", *common, "--tuned"]) == 0
    artifact = joblib.load(tmp_path / "models" / "lightgbm.joblib")
    assert artifact["tuned_params"] == best


def test_tune_on_synthetic_data_saves_only_to_an_explicit_dir(
    tune_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = ["--config", str(tune_config), "--synthetic", N_TITLES]
    assert main(["tune", *common, "--model", "lightgbm", "--n-trials", "2"]) == 0
    out = capsys.readouterr().out
    assert "search of lightgbm: 2 trials" in out
    assert "saved" not in out
    assert not (tmp_path / "tuned").exists()

    assert main(["evaluate", *common, "--tuned", "--models", "lightgbm"]) == 0
    assert "lightgbm: no tuned parameters" in capsys.readouterr().out

    assert main(["tune", *common, "--model", "ridge"]) == 1
    assert "no search space for 'ridge'" in capsys.readouterr().err


@pytest.fixture
def interval_config(tmp_path: Path) -> Path:
    path = tmp_path / "intervals.yaml"
    path.write_text(
        "seed: 3\nsplit:\n  cv_folds: 3\n  min_train_years: 5\n"
        "intervals:\n  methods: [quantile, conformal]\n  point_model: ridge\n",
        encoding="utf-8",
    )
    return path


def test_intervals_compares_methods_and_writes_reports(
    interval_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = importlib.util.find_spec

    def find_spec(name: str, *args: object) -> object:
        return None if name == "lightgbm" else original(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    reports = tmp_path / "reports"
    argv = ["intervals", "--config", str(interval_config), "--synthetic", N_TITLES]
    assert main([*argv, "--report-dir", str(reports)]) == 0
    out = capsys.readouterr().out
    assert "skipped quantile: lightgbm is not installed" in out
    assert "cross-validated central 90% intervals" in out
    assert "coverage by release year:" in out
    folds = out.split("folds:")[1].split("cross-validated")[0]
    assert len(folds.strip().splitlines()) == 1 + 3

    record = json.loads((reports / "interval_metrics.json").read_text("utf-8"))
    assert list(record["methods"]) == ["conformal"]
    assert record["metadata"]["skipped_models"] == ["quantile"]
    markdown = (reports / "interval_comparison.md").read_text("utf-8")
    assert "not evaluated" in markdown
    has_figure = original("matplotlib") is not None
    assert (reports / "figures" / "interval_coverage.png").is_file() == has_figure
    assert ("## Reliability figure" in markdown) == has_figure

    assert main([*argv, "--methods", "quantile"]) == 1
    assert "unavailable" in capsys.readouterr().err


def test_intervals_without_matplotlib_skip_the_figure(
    interval_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = importlib.util.find_spec

    def find_spec(name: str, *args: object) -> object:
        return None if name == "matplotlib" else original(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    reports = tmp_path / "reports"
    argv = ["intervals", "--config", str(interval_config), "--synthetic", N_TITLES]
    assert main([*argv, "--methods", "conformal", "--report-dir", str(reports)]) == 0
    assert "skipped the reliability figure" in capsys.readouterr().out
    assert not (reports / "figures").exists()
    assert (reports / "interval_comparison.md").is_file()


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm is not installed")
def test_intervals_runs_every_default_method(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fast = "{n_estimators: 100, early_stopping_rounds: 10}"
    path = tmp_path / "fast.yaml"
    path.write_text(
        "seed: 3\nsplit:\n  cv_folds: 2\n  min_train_years: 5\n"
        f"models:\n  params:\n    lightgbm: {fast}\n"
        f"intervals:\n  quantile_params: {fast}\n",
        encoding="utf-8",
    )
    argv = ["intervals", "--config", str(path), "--synthetic", N_TITLES]
    assert main(argv) == 0
    table = capsys.readouterr().out.split("intervals (sales")[1]
    for method in ("quantile", "conformal", "cqr"):
        assert method in table


@pytest.fixture
def explain_config(tmp_path: Path) -> Path:
    path = tmp_path / "explain.yaml"
    path.write_text(
        "seed: 3\nmodels:\n  params:\n"
        "    lightgbm: {n_estimators: 60, early_stopping_rounds: 10}\n"
        "explain:\n  models: [lightgbm, catboost]\n  max_rows: 200\n  top: 5\n"
        "  dependence: 2\n  n_local: 2\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm is not installed")
def test_explain_writes_shap_importance_and_figures(
    explain_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("shap")
    original = importlib.util.find_spec

    def find_spec(name: str, *args: object) -> object:
        return None if name == "catboost" else original(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    reports = tmp_path / "reports"
    argv = ["explain", "--config", str(explain_config), "--synthetic", N_TITLES]
    assert main([*argv, "--report-dir", str(reports)]) == 0
    out = capsys.readouterr().out
    assert "skipped catboost: catboost is not installed" in out
    assert "mean |SHAP value| by input column on 200 of" in out
    assert "publisher" in out.split("scale):")[1]
    local_out = out.split("local explanations of lightgbm for 2 explained")[1]
    assert local_out.count("million units") == 2
    assert "other columns" in local_out

    record = json.loads((reports / "shap_importance.json").read_text("utf-8"))
    assert list(record["models"]) == ["lightgbm"]
    assert record["models"]["lightgbm"]["n_rows"] == 200
    assert record["metadata"]["skipped_models"] == ["catboost"]
    assert record["metadata"]["explained_rows"]["seed"] == 3
    markdown = (reports / "shap_importance.md").read_text("utf-8")
    assert "random subsample of 200" in markdown
    assert markdown.count("### ") == 2
    assert len(record["local_explanations"]) == 2
    first, second = record["local_explanations"]
    assert first["predicted_sales"] >= second["predicted_sales"]
    terms = sum(term["shap"] for term in first["contributions"])
    assert first["base_value"] + terms == pytest.approx(first["prediction"])
    assert record["metadata"]["local_model"] == "lightgbm"
    has_figures = original("matplotlib") is not None
    for kind in ("importance", "summary"):
        figure = reports / "figures" / f"shap_{kind}_lightgbm.png"
        assert figure.is_file() == has_figures
    assert ("## Figures" in markdown) == has_figures
    assert ("## Dependence plots" in markdown) == has_figures
    dependence = sorted((reports / "figures").glob("shap_dependence_lightgbm_*.png"))
    local_figures = sorted((reports / "figures").glob("shap_local_lightgbm_*.png"))
    assert len(dependence) == (2 if has_figures else 0)
    assert len(local_figures) == (2 if has_figures else 0)
    if has_figures:
        assert "console" not in record["metadata"]["dependence_figures"]

    assert main([*argv, "--models", "catboost"]) == 1
    assert "unavailable" in capsys.readouterr().err


def test_explain_rejects_models_without_trees(
    explain_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["explain", "--config", str(explain_config), "--synthetic", N_TITLES]
    assert main([*argv, "--models", "ridge"]) == 1
    assert "boosting models only" in capsys.readouterr().err


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm is not installed")
def test_explain_needs_shap_and_skips_figures_without_matplotlib(
    explain_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = importlib.util.find_spec
    missing = {"shap"}

    def find_spec(name: str, *args: object) -> object:
        return None if name in missing else original(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    reports = tmp_path / "reports"
    argv = ["explain", "--config", str(explain_config), "--synthetic", N_TITLES]
    argv += ["--models", "lightgbm", "--report-dir", str(reports)]
    assert main(argv) == 1
    assert "vgsales[explain]" in capsys.readouterr().err

    if original("shap") is None:
        return
    missing = {"matplotlib"}
    assert main(argv) == 0
    assert "skipped the SHAP figures" in capsys.readouterr().out
    assert not (reports / "figures").exists()
    assert "## Figures" not in (reports / "shap_importance.md").read_text("utf-8")


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm is not installed")
def test_explain_decomposes_the_listed_titles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("shap")
    config = load_config()
    raw = make_synthetic(SyntheticSpec(n_titles=int(N_TITLES)), seed=3)
    X, _ = prepare_dataset(raw, config).development()
    key = X["title_key"].iloc[0]
    n_rows = int((X["title_key"] == key).sum())
    path = tmp_path / "titles.yaml"
    path.write_text(
        "seed: 3\nmodels:\n  params:\n"
        "    lightgbm: {n_estimators: 40, early_stopping_rounds: 10}\n"
        "explain:\n  models: [lightgbm]\n  max_rows: 50\n  dependence: 0\n"
        f"  titles: ['{key.upper()}', 'No Such Game']\n",
        encoding="utf-8",
    )
    reports = tmp_path / "reports"
    argv = ["explain", "--config", str(path), "--synthetic", N_TITLES]
    assert main([*argv, "--report-dir", str(reports)]) == 0
    out = capsys.readouterr().out
    assert "title 'No Such Game' is not among the development rows" in out
    assert "for the development rows of the titles in explain.titles" in out
    record = json.loads((reports / "shap_importance.json").read_text("utf-8"))
    labels = [item["label"] for item in record["local_explanations"]]
    assert len(labels) == n_rows
    assert all(label.startswith(key) for label in labels)
    assert "dependence_figures" not in record["metadata"]


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm is not installed")
def test_explain_holdout_adds_permutation_importance(
    explain_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("shap")
    reports = tmp_path / "reports"
    argv = ["explain", "--config", str(explain_config), "--synthetic", N_TITLES]
    argv += ["--models", "lightgbm", "--report-dir", str(reports)]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "permutation importance" not in out
    markdown = (reports / "shap_importance.md").read_text("utf-8")
    assert "The holdout is not used." in markdown

    assert main([*argv, "--holdout"]) == 0
    out = capsys.readouterr().out
    assert "permutation importance of lightgbm on" in out
    assert "Spearman rank correlation:" in out
    record = json.loads((reports / "shap_importance.json").read_text("utf-8"))
    permutation = record["permutation"]
    assert permutation["model"] == "lightgbm"
    assert permutation["metric"] == "rmsle"
    assert permutation["n_repeats"] == 10
    groups = [row["group"] for row in permutation["importance"]]
    assert "critic_score" in groups
    assert "critic_missing" not in groups
    assert set(permutation["scores"]) == set(groups)
    markdown = (reports / "shap_importance.md").read_text("utf-8")
    assert "## Permutation importance on the holdout" in markdown
    assert "used only for the permutation importance" in markdown
