import json

import numpy as np
import pandas as pd
import pytest
import yaml

from vgsales.config import load_config
from vgsales.data.splits import describe_folds, expanding_window_folds
from vgsales.evaluation.reports import (
    tuning_markdown,
    tuning_record,
    write_tuning_report,
)
from vgsales.tuning.search import TRIAL_COLUMNS, TuningResult
from vgsales.tuning.spaces import get_space
from vgsales.tuning.store import (
    load_tuned,
    read_tuned,
    save_tuned,
    tuned_overrides,
    tuned_path,
)

BEST = {"learning_rate": 0.031, "num_leaves": 47, "min_child_samples": 12}


def _result(name: str = "lightgbm", params: dict | None = None) -> TuningResult:
    params = dict(BEST if params is None else params)
    years = np.repeat(np.arange(2000, 2012), 5)
    folds = expanding_window_folds(years, n_splits=3, min_train_years=5)
    trials = pd.DataFrame(
        {
            "number": [0, 1, 2],
            "state": ["complete", "pruned", "complete"],
            "value": [0.30, np.nan, 0.25],
            "partial_value": [0.30, 0.41, 0.25],
            "n_folds": [3, 1, 3],
            "seconds": [1.2, 0.3, 1.1],
            **{key: [value] * 3 for key, value in params.items()},
        }
    )
    return TuningResult(
        name=name,
        metric="rmsle",
        direction="minimize",
        best_params=params,
        best_value=0.25,
        best_fold_scores=(0.2, 0.3, 0.25),
        fixed_params={"n_estimators": 200},
        folds=describe_folds(folds),
        seed=7,
        trials=trials,
    )


def test_saved_parameters_load_back(tmp_path) -> None:
    path = save_tuned(_result(), tmp_path / "tuned", {"source": "synthetic"})
    assert path == tuned_path(tmp_path / "tuned", "lightgbm")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Tuned by `vgsales tune`")

    record = yaml.safe_load(text)
    assert record["model"] == "lightgbm"
    assert record["objective"]["fold_scores"] == [0.2, 0.3, 0.25]
    assert record["search"] == {
        "n_trials": 3,
        "n_complete": 2,
        "n_pruned": 1,
        "seed": 7,
        "fixed_params": {"n_estimators": 200},
    }
    assert record["metadata"]["source"] == "synthetic"
    assert "created" in record["metadata"]
    assert load_tuned("lightgbm", tmp_path / "tuned") == BEST


def test_loaded_parameters_are_validated(tmp_path) -> None:
    save_tuned(_result("xgboost", {"max_depth": 4}), tmp_path)
    tuned_path(tmp_path, "xgboost").rename(tuned_path(tmp_path, "lightgbm"))
    with pytest.raises(ValueError, match="parameters of 'xgboost'"):
        load_tuned("lightgbm", tmp_path)

    save_tuned(_result(params={**BEST, "num_leaves": 4096}), tmp_path)
    with pytest.raises(ValueError, match="outside"):
        load_tuned("lightgbm", tmp_path)

    save_tuned(_result(params={**BEST, "max_depth": 4}), tmp_path)
    with pytest.raises(ValueError, match="outside the lightgbm search space"):
        load_tuned("lightgbm", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_tuned("catboost", tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [("[1, 2]\n", "does not name a model"), ("model: lightgbm\n", "no params")],
)
def test_malformed_files_are_rejected(tmp_path, content: str, message: str) -> None:
    path = tmp_path / "lightgbm.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_tuned(path)


def test_overrides_cover_only_models_with_files(tmp_path) -> None:
    save_tuned(_result(), tmp_path)
    overrides = tuned_overrides(["ridge", "lightgbm", "xgboost"], tmp_path)
    assert overrides == {"lightgbm": BEST}
    assert tuned_overrides(["lightgbm"], tmp_path / "missing") == {}


def test_tuned_values_lie_in_the_spaces() -> None:
    space = get_space("lightgbm")
    assert all(space[key].contains(value) for key, value in BEST.items())
    assert load_config().output.tuned_dir == "configs/tuned"


def test_tuning_report(tmp_path) -> None:
    result = _result()
    metadata = {"source": "synthetic", "version": "0.1.0"}
    markdown = tuning_markdown(result, metadata)
    assert markdown.startswith("# Hyperparameter search: lightgbm")
    assert "The data are synthetic" in markdown
    assert "(sampler seed 7). Completed: 2; pruned by the median rule: 1." in markdown
    assert "n_estimators=200" in markdown
    assert "Fold mean RMSLE: 0.2500." in markdown
    completed = markdown.split("## Completed trials")[1]
    assert completed.index("| 2 | 0.2500") < completed.index("| 0 | 0.3000")
    assert "| 1 |" not in completed

    record = tuning_record(result, metadata)
    assert record["params"] == BEST
    assert record["objective"]["n_folds"] == 3

    json_path, md_path = write_tuning_report(result, tmp_path, metadata)
    assert md_path.name == "tuning_lightgbm.md"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert [trial["state"] for trial in saved["trials"]] == list(result.trials["state"])
    assert saved["trials"][1]["value"] is None
    assert set(TRIAL_COLUMNS) <= set(saved["trials"][0])
