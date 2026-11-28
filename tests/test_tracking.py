import sys
from pathlib import Path

import numpy as np
import pytest

from vgsales.config import TrackingConfig, load_config
from vgsales.tracking import (
    MAX_PARAM_LENGTH,
    MlflowTracker,
    Tracker,
    clean_key,
    finite_metrics,
    flatten_params,
    make_tracker,
    mlflow_available,
)


def test_flatten_params_uses_dotted_keys() -> None:
    params = {
        "seed": 3,
        "models": {"params": {"lightgbm": {"num_leaves": 31}, "ridge": {}}},
        "features": {"categorical": ("console", "genre")},
        "data": {"snapshot_date": None},
        "R^2 log": 0.5,
    }
    flat = flatten_params(params)
    assert flat == {
        "seed": "3",
        "models.params.lightgbm.num_leaves": "31",
        "models.params.ridge": "{}",
        "features.categorical": "['console', 'genre']",
        "data.snapshot_date": "None",
        "R_2 log": "0.5",
    }
    assert flatten_params({"a": 1}, prefix="tuned") == {"tuned.a": "1"}
    long = flatten_params({"x": "y" * (MAX_PARAM_LENGTH + 10)})["x"]
    assert len(long) == MAX_PARAM_LENGTH
    assert long.endswith("...")


def test_finite_metrics_drops_undefined_values() -> None:
    metrics = {
        "rmsle": 0.41,
        "r2_log": np.float64(0.2),
        "n_val": np.int64(120),
        "rmsle_std": float("nan"),
        "width": np.inf,
        "flag": True,
        "name": "fold 1",
    }
    assert finite_metrics(metrics) == {"rmsle": 0.41, "r2_log": 0.2, "n_val": 120.0}
    assert clean_key("coverage@90%") == "coverage_90_"


def test_disabled_tracking_records_nothing() -> None:
    tracker = make_tracker(TrackingConfig())
    assert type(tracker) is Tracker
    assert not tracker.enabled
    with tracker.run("cv", tags={"source": "synthetic"}) as active:
        assert active is tracker
        assert active.run_id is None
        active.log_params({"a": 1})
        active.log_metrics({"rmsle": 0.4}, step=0)
        active.set_tags({"model": "ridge"})
        active.log_artifact("missing.json")


def test_enabled_tracking_needs_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError, match=r"vgsales\[track\]"):
        make_tracker(TrackingConfig(enabled=True))


def test_tracking_section_defaults_and_validation() -> None:
    tracking = load_config().tracking
    assert tracking.enabled is False
    assert tracking.tracking_uri == "sqlite:///mlflow.db"
    assert tracking.experiment == "vgsales"
    assert tracking.artifact_location is None
    with pytest.raises(TypeError):
        TrackingConfig(enabled="yes")
    with pytest.raises(TypeError):
        TrackingConfig(experiment=" ")
    with pytest.raises(TypeError):
        TrackingConfig(tracking_uri="")
    with pytest.raises(TypeError):
        TrackingConfig(artifact_location=3)


@pytest.mark.skipif(not mlflow_available(), reason="mlflow is not installed")
def test_mlflow_tracker_logs_a_run(tmp_path: Path) -> None:
    import mlflow

    config = TrackingConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment="unit",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    tracker = make_tracker(config)
    assert isinstance(tracker, MlflowTracker)
    assert tracker.enabled
    artifact = tmp_path / "report.json"
    artifact.write_text("{}", encoding="utf-8")
    with tracker.run("cv lightgbm", tags={"source": "synthetic"}) as active:
        run_id = active.run_id
        active.log_params({"models": {"lightgbm": {"num_leaves": 31}}})
        active.log_metrics({"rmsle": 0.5, "rmsle_std": float("nan")}, step=0)
        active.log_metrics({"rmsle": 0.4}, step=1)
        active.log_metrics({"only": float("nan")})
        active.set_tags({"model": "lightgbm"})
        active.log_artifact(artifact, directory="reports")
    assert run_id is not None
    assert tracker.run_id is None

    client = mlflow.MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(run_id)
    assert run.info.run_name == "cv lightgbm"
    assert run.data.params == {"models.lightgbm.num_leaves": "31"}
    assert run.data.metrics == {"rmsle": 0.4}
    history = client.get_metric_history(run_id, "rmsle")
    assert [(m.step, m.value) for m in history] == [(0, 0.5), (1, 0.4)]
    assert run.data.tags["source"] == "synthetic"
    assert run.data.tags["model"] == "lightgbm"
    paths = [item.path for item in client.list_artifacts(run_id, "reports")]
    assert paths == ["reports/report.json"]
    assert (tmp_path / "artifacts").is_dir()

    again = make_tracker(config)
    assert again.experiment_id == tracker.experiment_id
