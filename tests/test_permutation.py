import numpy as np
import pandas as pd
import pytest

from vgsales.config import load_config
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.explain.permutation import (
    LINKED_COLUMNS,
    PermutationResult,
    column_groups,
    compare_rankings,
    grouped_shap_importance,
    permutation_importance,
    rank_correlation,
)
from vgsales.explain.tree_shap import ShapExplanation
from vgsales.models.registry import is_available, make_model
from vgsales.models.target import with_target_transform


class AgeModel:
    """Predicts sales from the platform age alone."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(0.1 * X["platform_age_years"].to_numpy(dtype=float))


@pytest.fixture(scope="module")
def holdout() -> tuple[pd.DataFrame, np.ndarray]:
    """Holdout features with sales that the platform age determines exactly."""
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=900), seed=5), load_config()
    )
    X, _ = dataset.holdout()
    return X, AgeModel().predict(X)


def test_column_groups_join_linked_columns() -> None:
    columns = [
        "console",
        "release_month",
        "critic_score",
        "release_quarter",
        "critic_missing",
        "n_platforms",
    ]
    groups = column_groups(columns)
    assert groups == {
        "console": ("console",),
        "release_month": ("release_month", "release_quarter"),
        "critic_score": ("critic_score", "critic_missing"),
        "n_platforms": ("n_platforms",),
    }
    assert column_groups(["genre"], {}) == {"genre": ("genre",)}
    assert set(LINKED_COLUMNS["critic_score"]) == {"critic_score", "critic_missing"}
    with pytest.raises(ValueError, match="linked more than once"):
        column_groups(columns, {"a": ["console"], "b": ["console"]})


def test_only_the_used_column_matters(holdout) -> None:
    X, y = holdout
    result = permutation_importance(AgeModel(), X, y, n_repeats=4, random_state=1)
    assert isinstance(result, PermutationResult)
    assert "title_key" not in result.groups
    assert result.groups["critic_score"] == ("critic_score", "critic_missing")
    assert result.scores.shape == (len(result.groups), 4)
    table = result.importance()
    assert table.index[0] == "platform_age_years"
    assert result.baseline == pytest.approx(0.0)
    assert table.loc["platform_age_years", "importance"] > 0
    others = table.drop(index="platform_age_years")["importance"]
    assert np.allclose(others, 0.0)
    assert table.loc["release_month", "columns"] == (
        "release_month, release_quarter, holiday_season"
    )

    again = permutation_importance(AgeModel(), X, y, n_repeats=4, random_state=1)
    pd.testing.assert_frame_equal(again.scores, result.scores)


def test_greater_is_better_scores_flip_sign(holdout) -> None:
    X, y = holdout
    result = permutation_importance(
        AgeModel(), X, y, groups={"age": ["platform_age_years"]}, metric="r2_log"
    )
    assert result.scores.loc["age"].lt(result.baseline).all()
    assert result.importance().loc["age", "importance"] > 0
    single = permutation_importance(
        AgeModel(), X, y, groups={"age": ["platform_age_years"]}, n_repeats=1
    )
    assert np.isnan(single.importance().loc["age", "std"])


def test_invalid_arguments(holdout) -> None:
    X, y = holdout
    with pytest.raises(ValueError, match="metric"):
        permutation_importance(AgeModel(), X, y, metric="mape")
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_importance(AgeModel(), X, y, n_repeats=0)
    with pytest.raises(ValueError, match="at least two rows"):
        permutation_importance(AgeModel(), X.iloc[:1], y[:1])
    with pytest.raises(ValueError, match="not in X"):
        permutation_importance(AgeModel(), X, y, groups={"x": ["metacritic"]})


def test_grouped_shap_importance_sums_within_groups() -> None:
    values = pd.DataFrame(
        {
            "critic_score": [0.3, -0.2],
            "critic_missing": [-0.3, 0.1],
            "console_PS4": [0.1, 0.0],
            "console_PC": [0.0, -0.4],
        }
    )
    sources = {
        "critic_score": "critic_score",
        "critic_missing": "critic_missing",
        "console_PS4": "console",
        "console_PC": "console",
    }
    explanation = ShapExplanation(
        values=values, data=values, base_value=0.0, sources=sources
    )
    groups = {
        "critic_score": ("critic_score", "critic_missing"),
        "console": ("console",),
        "genre": ("genre",),
    }
    importance = grouped_shap_importance(explanation, groups)
    assert list(importance.index) == ["console", "critic_score"]
    assert importance["critic_score"] == pytest.approx(0.05)
    assert importance["console"] == pytest.approx(0.25)


def test_rank_comparison_and_correlation() -> None:
    permutation = pd.Series({"a": 0.3, "b": 0.2, "c": 0.1, "d": 0.0})
    table = compare_rankings(permutation, permutation * 2)
    assert list(table.columns) == [
        "permutation",
        "permutation_rank",
        "shap",
        "shap_rank",
    ]
    assert table["permutation_rank"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert rank_correlation(table) == pytest.approx(1.0)
    reversed_shap = pd.Series({"a": 0.0, "b": 0.1, "c": 0.2, "d": 0.3, "e": 1.0})
    table = compare_rankings(permutation, reversed_shap)
    assert list(table.index) == ["a", "b", "c", "d"]
    assert rank_correlation(table) == pytest.approx(-1.0)
    assert np.isnan(rank_correlation(table.iloc[:2]))
    flat = compare_rankings(permutation, pd.Series(0.5, index=permutation.index))
    assert np.isnan(rank_correlation(flat))


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm is not installed")
def test_permutation_importance_of_a_boosting_model() -> None:
    config = load_config()
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=900), seed=5), config
    )
    X, y = dataset.development()
    X_test, y_test = dataset.holdout()
    model = make_model("lightgbm", config, n_estimators=60, early_stopping_rounds=10)
    model = with_target_transform(model, "log1p").fit(X, y)
    result = permutation_importance(model, X_test, y_test, n_repeats=3)
    pred = model.predict(X_test)
    table = result.importance()
    assert set(table.index) == set(result.groups)
    assert np.isfinite(table["importance"]).all()
    assert result.baseline == pytest.approx(
        np.sqrt(np.mean((np.log1p(pred.clip(0)) - np.log1p(y_test)) ** 2))
    )
