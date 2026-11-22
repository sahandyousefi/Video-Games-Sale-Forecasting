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
        f"output:\n  model_dir: {tmp_path / 'models'}\n",
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
