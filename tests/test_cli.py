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
