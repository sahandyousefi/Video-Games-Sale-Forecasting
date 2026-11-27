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
from vgsales.evaluation.coverage import coverage_table
from vgsales.evaluation.cv import CVResult, cross_validate
from vgsales.evaluation.interval_cv import compare_intervals
from vgsales.evaluation.reports import (
    CV_JSON,
    CV_MARKDOWN,
    INTERVALS_JSON,
    INTERVALS_MARKDOWN,
    SERIES_JSON,
    SERIES_MARKDOWN,
    SYNTHETIC,
    coverage_pivot,
    cv_markdown,
    cv_record,
    interval_markdown,
    markdown_table,
    series_markdown,
    to_jsonable,
    write_cv_report,
    write_holdout_report,
    write_interval_report,
    write_series_report,
)
from vgsales.evaluation.rolling import rolling_origin_evaluation, rolling_origins
from vgsales.features.pipeline import make_preprocessor
from vgsales.models.conformal import SplitConformalRegressor
from vgsales.models.forecasting import (
    ExponentialSmoothingForecaster,
    SeasonalNaiveForecaster,
)


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


def test_series_report(tmp_path: Path) -> None:
    index = pd.date_range("2000-01-01", periods=72, freq="MS")
    rng = np.random.default_rng(1)
    wide = pd.DataFrame(
        rng.gamma(2.0, 0.5, size=(72, 3)), index=index, columns=["a", "b", "c"]
    )
    models = {
        "seasonal_naive": SeasonalNaiveForecaster(6),
        "ses": ExponentialSmoothingForecaster(6, seasonal=False),
    }
    result = rolling_origin_evaluation(
        wide, models, rolling_origins(index, 6, 3, step=6)
    )
    metadata = {
        "source": SYNTHETIC,
        "version": "0.1.0",
        "series": {
            "snapshot": "2006-06-30",
            "first_month": "2000-01",
            "last_month": "2005-12",
            "n_months": 72,
            "n_segments": 3,
            "n_titles": 400,
        },
    }
    json_path, md_path = write_series_report(
        result, tmp_path, "seasonal_naive", metadata
    )
    assert (json_path.name, md_path.name) == (SERIES_JSON, SERIES_MARKDOWN)
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["reference"] == "seasonal_naive"
    assert record["summary"]["seasonal_naive"]["skill"] == 0.0
    assert len(record["by_step"]) == 2 * 6
    assert record["paired_tests"]["tests"]["ses"]["n_origins"] == 3
    markdown = md_path.read_text(encoding="utf-8")
    assert "3 segments, monthly from 2000-01 to 2005-12" in markdown
    assert "## Paired comparison over origins" in markdown
    assert markdown == series_markdown(result, "seasonal_naive", record["metadata"])


@pytest.fixture(scope="module")
def interval_setup(setup):
    _, metadata = setup
    config = load_config()
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=500), seed=4), config
    )
    X, y = dataset.development()
    ridge = make_pipeline(make_preprocessor(scale=True), Ridge())
    models = {
        "conformal": SplitConformalRegressor(ridge, alpha=0.2),
        "wide": SplitConformalRegressor(ridge, alpha=0.2, calibration_years=1),
    }
    cv = ExpandingWindowSplit(n_splits=3, min_train_years=6)
    results = compare_intervals(models, X, y, cv, alpha=0.2)
    by_year = coverage_table(results, X["release_year"])
    by_genre = coverage_table(results, X["genre"])
    return results, by_year, by_genre, metadata


def test_coverage_pivot(interval_setup) -> None:
    results, by_year, _, _ = interval_setup
    wide = coverage_pivot(by_year, "release_year")
    assert list(wide.columns) == ["n", "conformal", "wide"]
    assert wide.index.name == "release_year"
    assert wide["n"].sum() == len(results["conformal"].predictions)
    first = by_year[by_year["method"] == "conformal"].set_index("release_year")
    assert np.allclose(wide["conformal"], first["coverage"])


def test_interval_report(interval_setup, tmp_path: Path) -> None:
    results, by_year, by_genre, metadata = interval_setup
    metadata = {**metadata, "figure": "figures/interval_coverage.png"}
    json_path, md_path = write_interval_report(
        results, by_year, by_genre, tmp_path, metadata
    )
    assert json_path == tmp_path / INTERVALS_JSON
    assert md_path == tmp_path / INTERVALS_MARKDOWN
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert list(record["methods"]) == ["conformal", "wide"]
    assert record["methods"]["conformal"]["alpha"] == 0.2
    assert len(record["methods"]["conformal"]["per_fold"]) == 3
    pooled = record["methods"]["conformal"]["summary"]["pooled"]
    assert pooled["coverage"] == pytest.approx(
        results["conformal"].summary().loc["pooled", "coverage"]
    )
    rows = record["coverage_by_genre"]
    assert {"method", "genre", "coverage_low", "coverage_high"} <= set(rows[0])
    assert "created" in record["metadata"]

    markdown = md_path.read_text(encoding="utf-8")
    for heading in (
        "# Prediction interval comparison",
        "## Methods",
        "## Folds",
        "## Scores",
        "## Coverage by release year",
        "## Coverage by genre",
        "## Reliability figure",
    ):
        assert heading in markdown
    assert "nominal level 80% (alpha = 0.2)" in markdown
    assert "- conformal: split conformal interval" in markdown
    assert "(point model: lightgbm)" in markdown
    assert "- wide\n" in markdown
    assert "The data are synthetic" in markdown
    assert "![Coverage by release year and genre]" in markdown

    without = interval_markdown(results, by_year, by_genre, {"source": "x.csv"})
    assert "Reliability figure" not in without
    assert "synthetic" not in without
    with pytest.raises(ValueError, match="must not be empty"):
        interval_markdown({}, by_year, by_genre)
