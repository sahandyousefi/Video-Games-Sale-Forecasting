import numpy as np
import pandas as pd
import pytest

from vgsales.config import load_config
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.dataset import prepare_dataset
from vgsales.explain.local import (
    LocalExplanation,
    explain_row,
    plot_local_explanation,
    plot_shap_dependence,
    row_label,
)
from vgsales.explain.tree_shap import ShapExplanation, explain_trees
from vgsales.models.registry import is_available, make_model
from vgsales.models.target import with_target_transform


def toy() -> tuple[ShapExplanation, pd.DataFrame]:
    index = pd.Index([3, 5, 8])
    values = pd.DataFrame(
        {
            "console": [0.30, -0.10, 0.02],
            "publisher_freq": [0.05, 0.20, -0.01],
            "publisher_te": [-0.15, 0.10, 0.00],
            "critic_score": [0.01, -0.40, 0.03],
            "release_year": [0.02, 0.00, -0.05],
        },
        index=index,
    )
    data = pd.DataFrame(
        {
            "console": [np.nan, np.nan, np.nan],
            "publisher_freq": [0.1, 0.3, 0.2],
            "publisher_te": [0.5, 0.2, 0.4],
            "critic_score": [np.nan, 8.5, 7.0],
            "release_year": [2004.0, 2010.0, 2010.0],
        },
        index=index,
    )
    sources = {
        "console": "console",
        "publisher_freq": "publisher",
        "publisher_te": "publisher",
        "critic_score": "critic_score",
        "release_year": "release_year",
    }
    explanation = ShapExplanation(
        values=values,
        data=data,
        base_value=0.2,
        sources=sources,
        categorical=("console", "genre"),
    )
    X = pd.DataFrame(
        {
            "console": ["PS2", "Wii", "DS"],
            "publisher": ["Nintendo", "Sega", "Capcom"],
            "release_year": [2004, 2010, 2010],
            "critic_score": [np.nan, 8.5, 7.0],
            "title_key": ["game a", "game b", "game c"],
        },
        index=index,
    )
    return explanation, X


def test_row_label_uses_title_console_and_year() -> None:
    _, X = toy()
    assert row_label(X.loc[3]) == "game a (PS2, 2004)"
    assert row_label(X.loc[5, ["title_key"]]) == "game b"
    assert row_label(X.loc[5, ["console"]]) == "5"


def test_explain_row_is_exact_and_ranked() -> None:
    explanation, X = toy()
    local = explain_row(explanation, X, 5, top=2)
    assert isinstance(local, LocalExplanation)
    assert local.label == "game b (Wii, 2010)"
    table = local.contributions
    assert list(table.index) == ["critic_score", "publisher", "2 other columns"]
    assert list(table["value"]) == ["8.5", "Sega", ""]
    assert table["shap"].tolist() == pytest.approx([-0.40, 0.30, -0.10])
    assert local.prediction == pytest.approx(0.2 + explanation.values.loc[5].sum())
    assert local.predicted_sales() == pytest.approx(np.expm1(local.prediction))

    full = explain_row(explanation, X, 3, top=10, transform="identity", label="x")
    assert full.label == "x"
    assert len(full.contributions) == 4
    assert full.contributions.loc["critic_score", "value"] == "missing"
    assert full.contributions.loc["console", "value"] == "PS2"
    assert full.predicted_sales() == pytest.approx(full.prediction)

    model_only = explain_row(explanation, X.drop(columns="critic_score"), 8)
    assert model_only.contributions.loc["critic_score", "value"] == "7"


def test_explain_row_rejects_unknown_rows() -> None:
    explanation, X = toy()
    with pytest.raises(ValueError, match="not among the explained rows"):
        explain_row(explanation, X, 99)
    with pytest.raises(ValueError, match="top must be positive"):
        explain_row(explanation, X, 3, top=0)


def test_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    explanation, X = toy()
    local = explain_row(explanation, X, 5, top=3)
    path = tmp_path / "local.png"
    fig = plot_local_explanation(local, path=path)
    assert path.is_file()
    ax = fig.axes[0]
    assert len(ax.patches) == len(local.contributions)
    assert ax.get_yticklabels()[-1].get_text() == "critic_score = 8.5"
    assert "game b" in ax.get_title(loc="left")

    path = tmp_path / "dependence.png"
    fig = plot_shap_dependence(
        explanation, "critic_score", color_by="release_year", path=path
    )
    assert path.is_file()
    assert "(1 missing not shown)" in fig.axes[0].get_xlabel()
    assert len(fig.axes) == 2
    plain = plot_shap_dependence(explanation, "publisher_te", max_points=2)
    assert len(plain.axes) == 1
    with pytest.raises(ValueError, match="nominal"):
        plot_shap_dependence(explanation, "console")
    with pytest.raises(ValueError, match="nominal"):
        plot_shap_dependence(explanation, "release_year", color_by="console")
    with pytest.raises(ValueError, match="unknown model column"):
        plot_shap_dependence(explanation, "genre_Action")


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm is not installed")
def test_local_explanation_matches_the_fitted_model() -> None:
    pytest.importorskip("shap")
    config = load_config()
    dataset = prepare_dataset(
        make_synthetic(SyntheticSpec(n_titles=600), seed=4), config
    )
    X, y = dataset.development()
    model = make_model("lightgbm", config, n_estimators=60, early_stopping_rounds=10)
    model = with_target_transform(model, "log1p").fit(X, y)
    explanation = explain_trees(model, X, max_rows=None)
    index = X.index[7]
    local = explain_row(explanation, X, index, top=4)
    assert local.prediction == pytest.approx(
        model.regressor_.predict(X.loc[[index]])[0], abs=1e-6
    )
    assert local.predicted_sales() == pytest.approx(
        model.predict(X.loc[[index]])[0], abs=1e-6
    )
    assert local.contributions.loc["console", "value"] == X.loc[index, "console"]
    assert local.label.startswith(str(X.loc[index, "title_key"]))
