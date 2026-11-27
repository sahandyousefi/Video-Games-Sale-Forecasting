import sys

import numpy as np
import pandas as pd
import pytest

from vgsales.config import FeatureConfig, load_config
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.explain.tree_shap import (
    ShapExplanation,
    explain_trees,
    importance_table,
    plot_shap_importance,
    plot_shap_summary,
    source_columns,
)
from vgsales.models.baselines import make_baseline
from vgsales.models.registry import is_available, make_model
from vgsales.models.target import with_target_transform

FAST = {"n_estimators": 60, "early_stopping_rounds": 10}
BOOSTING = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            not is_available(name), reason=f"{name} is not installed"
        ),
    )
    for name in ("xgboost", "lightgbm", "catboost")
]


def toy_explanation() -> ShapExplanation:
    values = pd.DataFrame(
        {
            "console_PS4": [0.2, -0.1, 0.0],
            "console_PC": [-0.3, 0.1, 0.0],
            "publisher_freq": [0.1, 0.1, -0.2],
            "publisher_te": [0.0, 0.2, 0.2],
            "critic_score": [0.5, -0.5, 0.1],
        },
        index=[7, 8, 9],
    )
    sources = {
        "console_PS4": "console",
        "console_PC": "console",
        "publisher_freq": "publisher",
        "publisher_te": "publisher",
        "critic_score": "critic_score",
    }
    return ShapExplanation(
        values=values,
        data=values.abs(),
        base_value=0.25,
        sources=sources,
        categorical=("console",),
    )


def test_source_columns_follow_the_longest_prefix() -> None:
    config = FeatureConfig()
    names = [
        "console_PS4",
        "console",
        "genre_Role-Playing",
        "publisher_freq",
        "developer_te",
        "release_year",
        "release_month",
        "critic_missing",
        "unrelated",
    ]
    mapping = source_columns(names, config)
    assert mapping == {
        "console_PS4": "console",
        "console": "console",
        "genre_Role-Playing": "genre",
        "publisher_freq": "publisher",
        "developer_te": "developer",
        "release_year": "release_year",
        "release_month": "release_month",
        "critic_missing": "critic_missing",
        "unrelated": "unrelated",
    }


def test_grouped_importance_sums_before_the_absolute_value() -> None:
    explanation = toy_explanation()
    assert np.allclose(explanation.predictions(), 0.25 + np.array([0.5, -0.2, 0.1]))
    importance = explanation.importance()
    assert importance.index[0] == "critic_score"
    assert importance.is_monotonic_decreasing
    grouped = explanation.grouped_values()
    assert list(grouped.columns) == ["console", "publisher", "critic_score"]
    assert np.allclose(grouped["console"], [-0.1, 0.0, 0.0])
    assert np.allclose(grouped.sum(axis=1), explanation.values.sum(axis=1))
    by_input = explanation.grouped_importance()
    assert by_input["console"] == pytest.approx(0.1 / 3)
    assert by_input["publisher"] == pytest.approx(0.4 / 3)
    assert by_input.name == "mean_abs_shap"


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=700), seed=9), load_config()
    )
    return dataset.development()


@pytest.mark.parametrize("name", BOOSTING)
def test_shap_values_add_up_to_the_prediction(name: str, data) -> None:
    pytest.importorskip("shap")
    X, y = data
    config = load_config()
    model = with_target_transform(make_model(name, config, **FAST), "log1p")
    model.fit(X, y)
    explanation = explain_trees(model, X, max_rows=None)
    assert explanation.values.index.equals(X.index)
    assert list(explanation.data.columns) == list(explanation.values.columns)
    z_hat = model.regressor_.predict(X)
    assert np.allclose(explanation.predictions(), z_hat, atol=1e-4)
    inputs = set(explanation.sources.values())
    features = config.features
    assert inputs == {
        *features.categorical,
        *features.high_cardinality,
        "release_year",
        *features.numeric,
    }
    assert set(explanation.grouped_importance().index) == inputs
    assert explanation.categorical == config.features.categorical

    sample = explain_trees(model.regressor_, X, max_rows=50, random_state=1)
    assert len(sample.values) == 50
    assert sample.values.index.isin(X.index).all()


def test_only_boosting_models_are_explained(data, monkeypatch) -> None:
    X, y = data
    ridge = make_baseline("ridge", load_config().features).fit(X, np.log1p(y))
    with pytest.raises(TypeError, match="gradient-boosting"):
        explain_trees(ridge, X)
    with pytest.raises(TypeError, match="DataFrame"):
        explain_trees(ridge, X.to_numpy())
    if is_available("lightgbm"):
        model = make_model("lightgbm", load_config(), **FAST).fit(X, np.log1p(y))
        monkeypatch.setitem(sys.modules, "shap", None)
        with pytest.raises(ImportError, match="vgsales\\[explain\\]"):
            explain_trees(model, X)


def test_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    explanation = toy_explanation()
    bars = plot_shap_importance(explanation, top=2, path=tmp_path / "bars.png")
    assert (tmp_path / "bars.png").stat().st_size > 0
    labels = [t.get_text() for t in bars.axes[0].get_yticklabels()]
    assert labels == ["publisher", "critic_score"]
    summary = plot_shap_summary(explanation, top=3, path=tmp_path / "summary.png")
    assert (tmp_path / "summary.png").stat().st_size > 0
    ticks = [t.get_text() for t in summary.axes[0].get_yticklabels()]
    assert ticks == list(explanation.importance().head(3).index[::-1])
    flat = plot_shap_importance(explanation, grouped=False, top=5)
    assert len(flat.axes[0].patches) == 5


def test_importance_table() -> None:
    explanation = toy_explanation()
    table = importance_table({"a": explanation, "b": explanation})
    assert list(table.columns) == ["a", "b"]
    assert table.index.name == "feature"
    assert table["a"].is_monotonic_decreasing
    assert importance_table({}).empty
