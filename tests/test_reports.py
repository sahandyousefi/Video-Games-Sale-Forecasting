import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from vgsales.config import load_config
from vgsales.data.splits import ExpandingWindowSplit, expanding_window_folds
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.evaluation.cv import CVResult, cross_validate
from vgsales.evaluation.reports import (
    CV_JSON,
    CV_MARKDOWN,
    cv_markdown,
    cv_record,
    markdown_table,
    to_jsonable,
    write_cv_report,
    write_holdout_report,
)
from vgsales.features.pipeline import make_preprocessor


@pytest.fixture(scope="module")
def setup() -> tuple[dict[str, CVResult], dict]:
    config = load_config()
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=500), seed=4), config
    )
    X, y = dataset.development()
    cv = ExpandingWindowSplit(n_splits=3, min_train_years=5)
    results = {
        "dummy": cross_validate(DummyRegressor(), X, y, cv),
        "ridge": cross_validate(
            make_pipeline(make_preprocessor(scale=True), Ridge()), X, y, cv
        ),
    }
    metadata = {
        "source": "synthetic",
        "version": "0.0.0",
        "config": config.to_dict(),
        "dataset": dataset.summary(),
    }
    return results, metadata


def test_markdown_table_layout() -> None:
    frame = pd.DataFrame(
        {"score": [0.123456, np.nan], "label": ["a|b", "c"]},
        index=pd.Index(["x", "y"], name="model"),
    )
    lines = markdown_table(frame).splitlines()
    assert lines[0] == "| model | score | label |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| x | 0.1235 | a\\|b |"
    assert lines[3] == "| y | n/a | c |"
    assert markdown_table(frame, index=False).splitlines()[0] == "| score | label |"


def test_to_jsonable_produces_strict_json() -> None:
    value = {
        "a": np.float64(1.5),
        "b": float("nan"),
        "c": np.array([1, 2]),
        "d": (np.int64(3), float("inf")),
        "e": Path("x/y"),
        1: np.bool_(True),
    }
    converted = to_jsonable(value)
    assert converted == {
        "a": 1.5,
        "b": None,
        "c": [1, 2],
        "d": [3, None],
        "e": "x/y",
        "1": True,
    }
    json.dumps(converted, allow_nan=False)


def test_cv_record_matches_results(setup: tuple[dict[str, CVResult], dict]) -> None:
    results, metadata = setup
    record = cv_record(results, metadata)
    assert set(record["models"]) == {"dummy", "ridge"}
    ridge = record["models"]["ridge"]
    assert len(ridge["per_fold"]) == 3
    expected = results["ridge"].summary()
    assert ridge["summary"]["pooled"]["rmsle"] == pytest.approx(
        expected.loc["pooled", "rmsle"]
    )
    assert record["metadata"]["source"] == "synthetic"
    paired = record["paired_tests"]
    assert paired["metric"] == "rmsle"
    (name,) = paired["tests"]
    assert paired["tests"][name]["n_folds"] == 3


def test_cv_markdown_content(setup: tuple[dict[str, CVResult], dict]) -> None:
    results, metadata = setup
    text = cv_markdown(results, metadata)
    assert text.startswith("# Cross-validation comparison\n")
    assert "The data are synthetic" in text
    assert "## Folds" in text
    assert "## Scores" in text
    ridge_line = next(line for line in text.splitlines() if line.startswith("| ridge"))
    pooled = results["ridge"].summary().loc["pooled", "rmsle"]
    assert f"{pooled:.4f}" in ridge_line
    assert "## Paired comparison" in text
    assert "smallest attainable p-value is 0.2500" in text
    with pytest.raises(ValueError, match="empty"):
        cv_markdown({})


def test_real_source_has_no_synthetic_note(
    setup: tuple[dict[str, CVResult], dict],
) -> None:
    results, metadata = setup
    text = cv_markdown(results, {**metadata, "source": "data/raw/vgchartz-2024.csv"})
    assert "synthetic" not in text


def test_write_cv_report(
    setup: tuple[dict[str, CVResult], dict], tmp_path: Path
) -> None:
    results, metadata = setup
    json_path, md_path = write_cv_report(results, tmp_path / "out", metadata)
    assert json_path == tmp_path / "out" / CV_JSON
    assert md_path == tmp_path / "out" / CV_MARKDOWN
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert "created" in record["metadata"]
    assert md_path.read_text(encoding="utf-8").endswith("|\n")


def test_undefined_spread_is_null_in_json(
    setup: tuple[dict[str, CVResult], dict], tmp_path: Path
) -> None:
    _, metadata = setup
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=300), seed=4), load_config()
    )
    X, y = dataset.development()
    folds = expanding_window_folds(X["release_year"], n_splits=1)
    single = {"dummy": cross_validate(DummyRegressor(), X, y, folds)}
    json_path, md_path = write_cv_report(single, tmp_path, metadata)
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["models"]["dummy"]["summary"]["std"]["rmsle"] is None
    assert "paired_tests" not in record
    text = md_path.read_text(encoding="utf-8")
    assert "(n/a)" in text
    assert "Paired comparison" not in text


def test_write_holdout_report(tmp_path: Path) -> None:
    scores = {
        "all": {"rmsle": 0.5, "r2": float("nan")},
        "uncensored": {"rmsle": 0.4, "r2": 0.2},
    }
    json_path, md_path = write_holdout_report(
        "ridge", scores, tmp_path, {"source": "data.csv"}
    )
    assert json_path.name == "holdout_ridge.json"
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["model"] == "ridge"
    assert record["scores"]["all"]["r2"] is None
    assert math.isclose(record["scores"]["uncensored"]["rmsle"], 0.4)
    text = md_path.read_text(encoding="utf-8")
    assert text.startswith("# Holdout scores: ridge")
    assert "| uncensored | 0.4000 | 0.2000 |" in text
