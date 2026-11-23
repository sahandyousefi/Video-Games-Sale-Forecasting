import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from vgsales.config import load_config
from vgsales.data.splits import ExpandingWindowSplit
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.evaluation.cv import CVResult, compare_models, cross_validate
from vgsales.evaluation.significance import (
    EXACT_MAX_N,
    PAIRED_COLUMNS,
    holm_adjust,
    paired_comparison,
    signed_rank_test,
)
from vgsales.features.pipeline import make_preprocessor


@pytest.fixture(scope="module")
def results() -> dict[str, CVResult]:
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=600), seed=11), load_config()
    )
    X, y = dataset.development()
    estimators = {
        "mean": DummyRegressor(strategy="mean"),
        "median": DummyRegressor(strategy="median"),
        "ridge": make_pipeline(make_preprocessor(scale=True), Ridge()),
    }
    cv = ExpandingWindowSplit(n_splits=4, min_train_years=5)
    return compare_models(estimators, X, y, cv)


@pytest.mark.parametrize("n", [3, 5, 8, 13])
def test_exact_p_values_match_scipy(n: int) -> None:
    rng = np.random.default_rng(n)
    for _ in range(20):
        d = rng.normal(0.3, 1.0, n)
        result = signed_rank_test(d)
        assert result.exact
        assert result.n == n
        assert result.p_value == pytest.approx(
            stats.wilcoxon(d, method="exact").pvalue, abs=1e-12
        )
        assert result.statistic == pytest.approx(np.sum(stats.rankdata(abs(d))[d > 0]))


def test_large_samples_use_the_normal_approximation() -> None:
    d = np.random.default_rng(1).normal(0.4, 1.0, EXACT_MAX_N + 10)
    result = signed_rank_test(d)
    assert not result.exact
    expected = stats.wilcoxon(d, method="approx", correction=False).pvalue
    assert result.p_value == pytest.approx(expected, rel=1e-10)


def test_floor_symmetry_zeros_and_ties() -> None:
    same_sign = signed_rank_test([0.1, 0.2, 0.3, 0.4, 0.5])
    assert same_sign.p_value == pytest.approx(0.0625)
    assert same_sign.p_min == pytest.approx(0.0625)
    assert same_sign.statistic == 15.0

    d = np.array([0.3, -0.1, 0.2, 0.0, 0.5, -0.4])
    result = signed_rank_test(d)
    flipped = signed_rank_test(-d)
    assert result.n == 5
    assert result.p_value == pytest.approx(flipped.p_value)
    assert result.statistic + flipped.statistic == pytest.approx(15.0)

    tied = signed_rank_test([1.0, 1.0, -1.0, 2.0, 2.0])
    # Midranks 2, 2, 2, 4.5, 4.5; 8 of 32 sign patterns deviate as much.
    assert tied.statistic == pytest.approx(13.0)
    assert tied.p_value == pytest.approx(8 / 32)

    none = signed_rank_test([0.0, 0.0])
    assert (none.n, none.p_value) == (0, 1.0)
    with pytest.raises(ValueError):
        signed_rank_test([])
    with pytest.raises(ValueError):
        signed_rank_test([1.0, np.nan])


def test_holm_adjustment() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.2])
    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.2])
    np.testing.assert_allclose(holm_adjust([0.5, 0.6]), [1.0, 1.0])
    assert holm_adjust([]).size == 0


def test_paired_comparison_against_the_best_model(
    results: dict[str, CVResult],
) -> None:
    table = paired_comparison(results)
    pooled = {name: r.summary().loc["pooled", "rmsle"] for name, r in results.items()}
    best = min(pooled, key=pooled.get)
    assert list(table.columns) == list(PAIRED_COLUMNS)
    assert set(table.index) == set(results) - {best}
    assert (table["reference"] == best).all()
    assert table["p_value"].is_monotonic_increasing
    assert (table["p_holm"] >= table["p_value"]).all()
    for name, row in table.iterrows():
        diff = (
            results[name].per_fold["rmsle"].to_numpy()
            - results[best].per_fold["rmsle"].to_numpy()
        )
        assert row["n_folds"] == 4
        assert row["n_better"] == int(np.sum(diff < 0))
        assert row["median_diff"] == pytest.approx(np.median(diff))
        assert row["p_value"] == pytest.approx(signed_rank_test(diff).p_value)
        assert row["p_min"] == pytest.approx(2.0**-3)


def test_greater_is_better_metrics_count_wins_upwards(
    results: dict[str, CVResult],
) -> None:
    table = paired_comparison(results, metric="r2_log", reference="mean")
    for name, row in table.iterrows():
        diff = (
            results[name].per_fold["r2_log"].to_numpy()
            - results["mean"].per_fold["r2_log"].to_numpy()
        )
        assert row["n_better"] == int(np.sum(diff > 0))


def test_paired_comparison_rejects_invalid_input(
    results: dict[str, CVResult],
) -> None:
    with pytest.raises(ValueError, match="at least two"):
        paired_comparison({"mean": results["mean"]})
    with pytest.raises(ValueError, match="unknown reference"):
        paired_comparison(results, reference="xgboost")
    with pytest.raises(ValueError, match="not computed"):
        paired_comparison(results, metric="pinball")
    mean = results["mean"]
    shifted = CVResult(
        mean.per_fold.assign(n_train=mean.per_fold["n_train"] + 1),
        mean.predictions,
        mean.metrics,
    )
    with pytest.raises(ValueError, match="same folds"):
        paired_comparison({"mean": mean, "shifted": shifted})


def test_identical_models_are_not_different() -> None:
    X = pd.DataFrame({"release_year": np.repeat(np.arange(2000, 2010), 5)})
    y = np.linspace(0.1, 2.0, len(X))
    folds = ExpandingWindowSplit(n_splits=3, min_train_years=3).folds(X)
    pair = {
        "a": cross_validate(DummyRegressor(), X, y, folds),
        "b": cross_validate(DummyRegressor(), X, y, folds),
    }
    row = paired_comparison(pair).iloc[0]
    assert row["median_diff"] == 0.0
    assert row["p_value"] == 1.0
