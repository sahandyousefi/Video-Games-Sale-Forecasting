import numpy as np
import pandas as pd
import pytest

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.loading import coerce_raw
from vgsales.data.profiling import (
    censoring_summary,
    coverage_by_year,
    missingness,
    target_summary,
    top_share,
)
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope


@pytest.fixture
def scoped() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "release_year": [2000, 2000, 2000, 2001, 2001],
            "total_sales": [1.0, np.nan, 3.0, np.nan, np.nan],
            "critic_score": [np.nan, 7.0, np.nan, np.nan, 8.0],
            "censored": [False, False, False, True, True],
        }
    )


def test_missingness_counts_and_order(scoped: pd.DataFrame) -> None:
    out = missingness(scoped)
    assert out.index.tolist() == [
        "total_sales",
        "critic_score",
        "release_year",
        "censored",
    ]
    assert out.loc["total_sales", "n_missing"] == 3
    assert out.loc["critic_score", "share_missing"] == pytest.approx(0.6)
    subset = missingness(scoped, columns=["release_year"])
    assert subset["n_missing"].tolist() == [0]


def test_coverage_by_year(scoped: pd.DataFrame) -> None:
    out = coverage_by_year(scoped)
    assert out.index.tolist() == [2000, 2001]
    assert out["n_rows"].tolist() == [3, 2]
    assert out["n_tracked"].tolist() == [2, 0]
    assert out["tracked_share"].tolist() == pytest.approx([2 / 3, 0.0])
    assert out.loc[2000, "median_sales"] == 2.0
    assert np.isnan(out.loc[2001, "median_sales"])
    assert out["n_censored"].tolist() == [0, 2]
    assert "n_censored" not in coverage_by_year(scoped.drop(columns="censored"))


def test_top_share() -> None:
    sales = pd.Series([10.0, 5.0, 3.0, 1.0, 1.0])
    assert top_share(sales, 0.2) == pytest.approx(0.5)
    assert top_share(sales, 0.01) == pytest.approx(0.5)
    assert top_share(sales, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="fraction"):
        top_share(sales, 0.0)
    with pytest.raises(ValueError, match="zero"):
        top_share(pd.Series([0.0, 0.0]), 0.5)


def test_target_summary_on_both_scales() -> None:
    sales = pd.Series([0.01, 0.05, 0.1, 0.2, 4.0])
    out = target_summary(sales)
    assert list(out.columns) == ["raw", "log1p"]
    assert out.loc["count", "raw"] == 5
    assert out.loc["q50", "raw"] == pytest.approx(0.1)
    assert out.loc["q50", "log1p"] == pytest.approx(np.log1p(0.1))
    assert out.loc["max", "log1p"] == pytest.approx(np.log1p(4.0))
    # The log transform reduces the right skew of the heavy-tailed sample.
    assert out.loc["skewness", "raw"] > out.loc["skewness", "log1p"] > 0


def test_target_summary_rejects_invalid_sales() -> None:
    with pytest.raises(ValueError, match="missing"):
        target_summary(pd.Series([1.0, np.nan]))
    with pytest.raises(ValueError, match="non-negative"):
        target_summary(pd.Series([1.0, -0.5]))
    with pytest.raises(ValueError, match="empty"):
        target_summary(pd.Series([], dtype=float))


def test_censoring_summary(scoped: pd.DataFrame) -> None:
    tracked = scoped.dropna(subset=["total_sales"])
    out = censoring_summary(tracked)
    assert out.index.tolist() == [False, True]
    assert out["n_rows"].tolist() == [2, 0]
    assert out.loc[False, "median_sales"] == 2.0
    assert out.loc[False, "mean_log_sales"] == pytest.approx(
        np.mean(np.log1p([1.0, 3.0]))
    )
    assert np.isnan(out.loc[True, "median_sales"])


def test_profiles_on_synthetic_extract() -> None:
    config = load_config()
    raw = make_synthetic(SyntheticSpec(n_titles=300), seed=8)
    loaded, _ = coerce_raw(raw)
    all_rows, _ = apply_release_scope(loaded, config.data)
    coverage = coverage_by_year(all_rows)
    assert coverage["n_rows"].sum() == len(all_rows)
    assert coverage["n_tracked"].sum() == all_rows["total_sales"].notna().sum()
    assert missingness(loaded).loc["total_sales", "share_missing"] > 0.3
    tracked, _ = apply_release_scope(clean(raw)[0], config.data)
    summary = target_summary(tracked["total_sales"])
    assert summary.loc["count", "raw"] == len(tracked)
    assert censoring_summary(tracked)["n_rows"].sum() == len(tracked)
