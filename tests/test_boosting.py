from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.schema import TARGET
from vgsales.data.splits import ExpandingWindowSplit, holdout_split
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.evaluation.cv import cross_validate
from vgsales.features.pipeline import make_preprocessor
from vgsales.features.prerelease import GROUP_COLUMN, build_prerelease_features
from vgsales.models.boosting import (
    BOOSTING_NAMES,
    LightGBMRegressor,
    TemporalBoostingRegressor,
    XGBoostRegressor,
    _require,
    make_boosting,
)


class RecordingBooster:
    """Predicts the training mean and records what it was fitted on."""

    def __init__(self, n_estimators: int, best: int) -> None:
        self.n_estimators = n_estimators
        self.best = best
        self.calls: list[dict[str, Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Any) -> None:
        self.calls.append({"n_train": len(X), "eval_set": eval_set})
        self.mean_ = float(np.mean(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_)


class RecordingRegressor(TemporalBoostingRegressor):
    best: int = 7

    def _make_preprocessor(self) -> Pipeline:
        return make_preprocessor(self.config)

    def _make_booster(self, n_estimators: int, early_stopping: bool) -> Any:
        self._booster_params({"n_estimators": n_estimators})
        return RecordingBooster(n_estimators, min(self.best, n_estimators))

    def _fit_booster(self, booster: Any, X: Any, y: Any, eval_set: Any) -> None:
        booster.fit(X, y, eval_set)

    def _best_n_estimators(self, booster: Any) -> int:
        return booster.best


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    raw = make_synthetic(SyntheticSpec(n_titles=800), seed=5)
    scoped, _ = apply_release_scope(clean(raw)[0], load_config().data)
    return build_prerelease_features(scoped), np.log1p(scoped[TARGET].to_numpy())


def test_inner_fold_holds_the_latest_years_with_purging(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    model = RecordingRegressor(validation_years=3, refit=False).fit(X, z)
    train_idx, val_idx = holdout_split(X["release_year"], 3, X[GROUP_COLUMN])
    last = int(X["release_year"].max())
    assert model.early_stopping_years_ == (last - 2, last)
    (call,) = model.booster_.calls
    assert call["n_train"] == train_idx.size
    X_val, z_val = call["eval_set"]
    assert len(X_val) == val_idx.size
    np.testing.assert_array_equal(z_val, z[val_idx])
    assert model.validation_rmse_ == pytest.approx(
        np.sqrt(np.mean((z_val - z[train_idx].mean()) ** 2))
    )


def test_refit_uses_all_rows_and_the_selected_rounds(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    model = RecordingRegressor(n_estimators=50).fit(X, z)
    assert model.n_estimators_ == 7
    assert model.booster_.n_estimators == 7
    (call,) = model.booster_.calls
    assert call == {"n_train": len(X), "eval_set": None}
    np.testing.assert_allclose(model.predict(X), z.mean())


def test_zero_validation_years_disables_early_stopping(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    model = RecordingRegressor(n_estimators=30, validation_years=0).fit(X, z)
    assert model.early_stopping_years_ is None
    assert np.isnan(model.validation_rmse_)
    assert model.n_estimators_ == 30
    assert model.booster_.calls[0]["eval_set"] is None


def test_too_few_years_falls_back_with_a_warning(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    last = X["release_year"].max()
    recent = (X["release_year"] >= last - 1).to_numpy()
    with pytest.warns(UserWarning, match="early stopping is skipped"):
        model = RecordingRegressor(validation_years=2).fit(X[recent], z[recent])
    assert model.early_stopping_years_ is None


def test_invalid_arguments_are_rejected(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    with pytest.raises(ValueError, match="must not set"):
        RecordingRegressor(params={"n_estimators": 5}).fit(X, z)
    with pytest.raises(ValueError, match="validation_years"):
        RecordingRegressor(validation_years=-1).fit(X, z)
    with pytest.raises(ValueError, match="learning_rate"):
        RecordingRegressor(learning_rate=0.0).fit(X, z)
    with pytest.raises(ValueError, match="release_year"):
        RecordingRegressor().fit(X.drop(columns="release_year"), z)
    with pytest.raises(ValueError, match="finite"):
        RecordingRegressor().fit(X, np.full(len(X), np.nan))


def test_leakage_columns_are_rejected(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    with pytest.raises(ValueError, match="na_sales"):
        RecordingRegressor().fit(X.assign(na_sales=0.0), z)


def test_factory_splits_wrapper_and_booster_arguments() -> None:
    model = make_boosting("xgboost", n_estimators=10, validation_years=1, max_depth=3)
    assert isinstance(model, XGBoostRegressor)
    assert (model.n_estimators, model.validation_years) == (10, 1)
    assert model.params == {"max_depth": 3}
    assert clone(model).get_params()["params"] == {"max_depth": 3}
    assert BOOSTING_NAMES == ("xgboost", "lightgbm")
    assert isinstance(make_boosting("lightgbm"), LightGBMRegressor)
    with pytest.raises(ValueError, match="unknown boosting model"):
        make_boosting("adaboost")


def test_missing_library_names_the_extra() -> None:
    with pytest.raises(ImportError, match=r"vgsales\[gbm\]"):
        _require("vgsales_missing_module")


def test_xgboost_stops_on_the_inner_fold(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    pytest.importorskip("xgboost")
    X, z = data
    model = make_boosting(
        "xgboost",
        n_estimators=400,
        learning_rate=0.3,
        early_stopping_rounds=10,
        refit=False,
    ).fit(X, z)
    curve = model.booster_.evals_result()["validation_0"]["rmse"]
    assert model.n_estimators_ < 400
    assert model.n_estimators_ == int(np.argmin(curve)) + 1
    assert model.validation_rmse_ == pytest.approx(min(curve), rel=1e-5)


def test_xgboost_refit_is_deterministic(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    pytest.importorskip("xgboost")
    X, z = data
    params = {"n_estimators": 200, "learning_rate": 0.2, "early_stopping_rounds": 10}
    first = make_boosting("xgboost", random_state=3, **params).fit(X, z)
    second = make_boosting("xgboost", random_state=3, **params).fit(X, z)
    assert first.booster_.get_booster().num_boosted_rounds() == first.n_estimators_
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_xgboost_under_cross_validation(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    pytest.importorskip("xgboost")
    X, z = data
    model = make_boosting(
        "xgboost", n_estimators=100, learning_rate=0.2, early_stopping_rounds=10
    )
    cv = ExpandingWindowSplit(n_splits=3, min_train_years=5, group_column=GROUP_COLUMN)
    result = cross_validate(model, X, np.expm1(z), cv)
    assert np.isfinite(result.per_fold["rmsle"]).all()
    assert len(result.predictions) == sum(f.val_idx.size for f in cv.folds(X))


def _split_kinds(model: LightGBMRegressor) -> set[tuple[int, str]]:
    def walk(node: dict[str, Any]) -> None:
        if "split_index" in node:
            kinds.add((node["split_feature"], node["decision_type"]))
            walk(node["left_child"])
            walk(node["right_child"])

    kinds: set[tuple[int, str]] = set()
    for tree in model.booster_.booster_.dump_model()["tree_info"]:
        walk(tree["tree_structure"])
    return kinds


def test_lightgbm_splits_console_and_genre_natively(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    pytest.importorskip("lightgbm")
    X, z = data
    model = make_boosting(
        "lightgbm",
        n_estimators=200,
        learning_rate=0.1,
        early_stopping_rounds=20,
        refit=False,
        min_child_samples=5,
        min_data_per_group=5,
    ).fit(X, z)
    curve = model.booster_.evals_result_["valid_0"]["rmse"]
    assert model.n_estimators_ == int(np.argmin(curve)) + 1
    assert model.validation_rmse_ == pytest.approx(min(curve), rel=1e-6)
    kinds = _split_kinds(model)
    # Columns 0 and 1 hold the console and genre codes.
    assert {feature for feature, kind in kinds if kind == "=="} <= {0, 1}
    assert any(kind == "==" for _, kind in kinds)


def test_lightgbm_handles_unseen_categories_and_missing_values(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    pytest.importorskip("lightgbm")
    X, z = data
    params = {"n_estimators": 100, "early_stopping_rounds": 10, "random_state": 1}
    model = make_boosting("lightgbm", **params).fit(X, z)
    again = make_boosting("lightgbm", **params).fit(X, z)
    assert model.booster_.n_estimators == model.n_estimators_
    np.testing.assert_array_equal(model.predict(X), again.predict(X))
    query = X.head(5).assign(console="unreleased console", critic_score=np.nan)
    assert np.isfinite(model.predict(query)).all()
