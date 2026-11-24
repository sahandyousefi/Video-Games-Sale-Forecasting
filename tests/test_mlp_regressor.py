import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from vgsales.config import config_from_dict, load_config  # noqa: E402
from vgsales.data.cleaning import clean  # noqa: E402
from vgsales.data.schema import TARGET  # noqa: E402
from vgsales.data.splits import ExpandingWindowSplit, holdout_split  # noqa: E402
from vgsales.data.synthetic import SyntheticSpec, make_synthetic  # noqa: E402
from vgsales.data.temporal import apply_release_scope  # noqa: E402
from vgsales.evaluation.cv import check_same_folds, compare_models  # noqa: E402
from vgsales.features.prerelease import (  # noqa: E402
    GROUP_COLUMN,
    build_prerelease_features,
)
from vgsales.models.neural.regressor import EmbeddingMLPRegressor  # noqa: E402
from vgsales.models.registry import make_model  # noqa: E402

SMALL = {"hidden": (16,), "max_epochs": 15, "patience": 3, "batch_size": 64}


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    raw = make_synthetic(SyntheticSpec(n_titles=800), seed=5)
    scoped, _ = apply_release_scope(clean(raw)[0], load_config().data)
    return build_prerelease_features(scoped), np.log1p(scoped[TARGET].to_numpy())


def test_epochs_are_selected_on_the_latest_years(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    model = EmbeddingMLPRegressor(validation_years=3, refit=False, **SMALL).fit(X, z)
    last = int(X["release_year"].max())
    assert model.early_stopping_years_ == (last - 2, last)
    history = model.selection_history_
    assert history is model.history_
    assert 1 <= model.best_epoch_ <= 15
    assert model.best_epoch_ == history.best_epoch
    assert model.validation_rmse_ == pytest.approx(np.sqrt(history.best_val_loss))

    _, val_idx = holdout_split(X["release_year"], 3, X[GROUP_COLUMN])
    residual = model.predict(X.iloc[val_idx]) - z[val_idx]
    assert np.sqrt(np.mean(residual**2)) == pytest.approx(
        model.validation_rmse_, rel=1e-5
    )


def test_refit_replays_the_selected_epochs_on_all_rows(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    model = EmbeddingMLPRegressor(**SMALL).fit(X, z)
    selection = model.selection_history_
    assert model.history_ is not selection
    assert model.history_.n_epochs == selection.best_epoch == model.best_epoch_
    assert model.history_.learning_rate == selection.learning_rate[: model.best_epoch_]
    assert model.history_.val_loss == []
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()
    assert not model.network_.training


def test_zero_validation_years_trains_for_max_epochs(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    params = {**SMALL, "max_epochs": 4}
    model = EmbeddingMLPRegressor(validation_years=0, **params).fit(X, z)
    assert model.selection_history_ is None
    assert model.early_stopping_years_ is None
    assert np.isnan(model.validation_rmse_)
    assert model.best_epoch_ == 4


def test_too_few_years_falls_back_with_a_warning(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    recent = (X["release_year"] >= X["release_year"].max() - 1).to_numpy()
    with pytest.warns(UserWarning, match="early stopping is skipped"):
        model = EmbeddingMLPRegressor(validation_years=2, **SMALL)
        model.fit(X[recent], z[recent])
    assert model.best_epoch_ == SMALL["max_epochs"]


def test_fits_are_reproducible_and_survive_pickling(
    data: tuple[pd.DataFrame, np.ndarray], tmp_path
) -> None:
    X, z = data
    first = EmbeddingMLPRegressor(random_state=4, **SMALL).fit(X, z)
    second = clone(first).fit(X, z)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    other = EmbeddingMLPRegressor(random_state=5, **SMALL).fit(X, z)
    assert not np.array_equal(first.predict(X), other.predict(X))

    path = tmp_path / "mlp.joblib"
    joblib.dump(first, path)
    np.testing.assert_array_equal(joblib.load(path).predict(X), first.predict(X))


def test_inputs_are_checked(data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, z = data
    with pytest.raises(ValueError, match="release_year"):
        EmbeddingMLPRegressor(**SMALL).fit(X.drop(columns="release_year"), z)
    with pytest.raises(ValueError, match="finite"):
        EmbeddingMLPRegressor(**SMALL).fit(X, np.full(len(X), np.nan))
    with pytest.raises(ValueError, match="validation_years"):
        EmbeddingMLPRegressor(validation_years=-1, **SMALL).fit(X, z)
    with pytest.raises(ValueError, match="na_sales"):
        EmbeddingMLPRegressor(**SMALL).fit(X.assign(na_sales=0.0), z)


def test_mlp_is_compared_on_the_shared_folds(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, z = data
    config = config_from_dict(
        {"models": {"evaluate": ["ridge", "mlp"], "params": {"mlp": SMALL}}}
    )
    models = {name: make_model(name, config) for name in config.models.evaluate}
    cv = ExpandingWindowSplit(n_splits=3, min_train_years=5)
    results = compare_models(models, X, np.expm1(z), cv, groups=X[GROUP_COLUMN])
    check_same_folds(results)
    scores = results["mlp"].per_fold["rmsle"]
    assert len(scores) == 3
    assert np.isfinite(scores).all()
