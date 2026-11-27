import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from vgsales.evaluation.coverage import (
    GROUP_COLUMNS,
    coverage_by,
    coverage_table,
    plot_coverage_reliability,
    wilson_interval,
)
from vgsales.evaluation.interval_cv import IntervalCVResult


def make_predictions() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.Index([10, 11, 12, 13, 14, 15, 16, 17], name="row")
    predictions = pd.DataFrame(
        {
            "fold": 0,
            "y_true": [0.5, 1.2, 0.1, 0.4, 1.0, 3.0, 0.2, 0.05],
            "y_pred": [0.5, 1.0, 0.2, 0.4, 1.0, 1.0, 0.2, 0.2],
            "lower": [0.2, 0.5, 0.15, 0.1, 0.5, 0.5, 0.1, 0.1],
            "upper": [1.0, 1.5, 0.4, 0.8, 2.0, 2.0, 0.4, 0.4],
        },
        index=index,
    )
    years = pd.Series(
        [2001, 2001, 2001, 2002, 2002, 2002, 2002, 2003],
        index=index,
        name="release_year",
    )
    return predictions, years


def test_wilson_interval_matches_scipy() -> None:
    covered = np.array([0, 3, 9, 10, 50])
    n = np.array([10, 10, 10, 10, 100])
    low, high = wilson_interval(covered, n, confidence=0.9)
    for k, trials, lo, hi in zip(covered, n, low, high, strict=True):
        reference = binomtest(int(k), int(trials)).proportion_ci(
            confidence_level=0.9, method="wilson"
        )
        assert lo == pytest.approx(reference.low)
        assert hi == pytest.approx(reference.high)
    assert (low <= covered / n).all()
    assert (covered / n <= high).all()
    assert high[-2] == 1.0
    assert low[0] == 0.0


def test_wilson_interval_validates_counts() -> None:
    with pytest.raises(ValueError, match="0 <= covered <= n"):
        wilson_interval([5], [4])
    with pytest.raises(ValueError, match="0 <= covered <= n"):
        wilson_interval([0], [0])
    with pytest.raises(ValueError, match="confidence"):
        wilson_interval([1], [2], confidence=1.0)
    with pytest.raises(ValueError, match="same shape"):
        wilson_interval([1, 2], [3])


def test_coverage_by_release_year() -> None:
    predictions, years = make_predictions()
    table = coverage_by(predictions, years, alpha=0.2)
    assert list(table.columns) == list(GROUP_COLUMNS)
    assert list(table.index) == [2001, 2002, 2003]
    assert table.index.name == "release_year"
    assert table["n"].tolist() == [3, 4, 1]
    # 2001: the third row lies below its interval; 2002: the sixth lies above.
    assert table["covered"].tolist() == [2, 3, 0]
    assert table.loc[2001, "below"] == pytest.approx(1 / 3)
    assert table.loc[2002, "above"] == pytest.approx(1 / 4)
    assert table.loc[2003, "below"] == pytest.approx(1.0)
    assert (table["coverage_low"] <= table["coverage"]).all()
    assert (table["coverage"] <= table["coverage_high"]).all()
    expected = np.mean(np.log1p([1.0, 1.5, 0.4]) - np.log1p([0.2, 0.5, 0.15]))
    assert table.loc[2001, "width_log"] == pytest.approx(expected)


def test_labels_must_cover_every_prediction() -> None:
    predictions, years = make_predictions()
    with pytest.raises(ValueError, match="missing for 2"):
        coverage_by(predictions, years.iloc[:-2], alpha=0.2)
    with pytest.raises(TypeError, match="Series"):
        coverage_by(predictions, years.to_numpy(), alpha=0.2)


def result_of(predictions: pd.DataFrame) -> IntervalCVResult:
    return IntervalCVResult(
        per_fold=pd.DataFrame(), predictions=predictions, alpha=0.2, metrics=()
    )


@pytest.fixture
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, years = make_predictions()
    wide = predictions.assign(lower=0.0, upper=5.0)
    results = {"quantile": result_of(predictions), "conformal": result_of(wide)}
    genres = pd.Series(
        ["Action", "Sports", "Action", "Sports", "Action", "Sports", "Action", "Misc"],
        index=predictions.index,
        name="genre",
    )
    return coverage_table(results, years), coverage_table(results, genres)


def test_coverage_table_is_long_by_method(tables) -> None:
    by_year, by_genre = tables
    assert list(by_year.columns) == ["method", "release_year", *GROUP_COLUMNS]
    assert by_year["method"].tolist() == ["quantile"] * 3 + ["conformal"] * 3
    wide = by_year[by_year["method"] == "conformal"]
    assert (wide["coverage"] == 1.0).all()
    assert set(by_genre["genre"]) == {"Action", "Misc", "Sports"}
    unnamed = coverage_table(
        {"a": result_of(make_predictions()[0])},
        make_predictions()[1].rename(None),
    )
    assert "group" in unnamed.columns


def test_reliability_figure(tables, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    by_year, by_genre = tables
    path = tmp_path / "coverage.png"
    figure = plot_coverage_reliability(
        by_year, by_genre, alpha=0.2, min_count=1, path=path
    )
    assert path.stat().st_size > 0
    year_ax, genre_ax = figure.axes
    assert year_ax.get_title(loc="left") == "Coverage by release year"
    assert [t.get_text() for t in genre_ax.get_yticklabels()] == [
        "Action",
        "Misc",
        "Sports",
    ]
    labels = [t.get_text() for t in figure.legends[0].get_texts()]
    assert labels == ["quantile", "conformal", "nominal 80%"]

    many = pd.concat(
        [by_year.assign(method=name) for name in ("a", "b", "c", "d")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="at most 3"):
        plot_coverage_reliability(many, by_genre, alpha=0.2)
