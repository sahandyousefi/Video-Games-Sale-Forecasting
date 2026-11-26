r"""Gradient-boosted tree regressors with temporal early stopping.

The number of boosting rounds :math:`M` is the main capacity parameter of a
boosted ensemble :math:`\hat z(x) = \sum_{m=1}^{M} \eta f_m(x)`, where
:math:`\eta` is the learning rate and :math:`f_m` the :math:`m`-th tree. Here
:math:`M` is chosen on an inner validation fold that respects time.

Let :math:`T` be the latest release year among the training rows and
:math:`v` the value of ``validation_years``. Rows released after
:math:`T - v` form the inner validation fold. Earlier rows whose title also
has a release in that fold are purged, as in
:func:`vgsales.data.splits.holdout_split`. The preprocessing pipeline and a
booster with at most ``n_estimators`` rounds are fitted on the remaining
rows, and boosting stops when the validation RMSE on the modelling scale has
not improved for ``early_stopping_rounds`` rounds. The selected number of
rounds :math:`M^*` is the one with the lowest validation RMSE. With
``refit=True`` the preprocessing and a booster with exactly :math:`M^*`
rounds are then refitted on all training rows.

Under cross-validation the estimator is fitted on the training rows of each
outer fold, so the inner validation fold lies inside the outer training
window and the outer validation years never influence :math:`M^*`. Under the
default ``log1p`` target transform the estimator receives
:math:`z = \log(1 + y)`, and the stopping criterion is the RMSLE of the
validation fold.

The boosting libraries are optional dependencies (``pip install
"vgsales[gbm]"``) and are imported when a model is fitted.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from collections.abc import Mapping
from types import ModuleType
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from vgsales.config import FeatureConfig
from vgsales.data.splits import holdout_split
from vgsales.data.temporal import RELEASE_YEAR
from vgsales.features.pipeline import make_native_preprocessor, make_preprocessor
from vgsales.features.prerelease import GROUP_COLUMN

BOOSTING_NAMES: Final = ("xgboost", "lightgbm", "catboost")

TIME_ORDER: Final = (RELEASE_YEAR, "release_month")

# Arguments of TemporalBoostingRegressor; the remaining factory arguments are
# passed to the booster of the library.
WRAPPER_PARAMS: Final = (
    "n_estimators",
    "learning_rate",
    "early_stopping_rounds",
    "validation_years",
    "refit",
    "random_state",
)


def _require(module: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            f"{module} is not installed; install the boosting extras with "
            "pip install 'vgsales[gbm]'"
        ) from error


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def chronological_order(X: pd.DataFrame) -> np.ndarray:
    """Return the positions of the rows sorted by release year and month.

    The sort is stable, so rows of the same month keep their input order.

    Raises
    ------
    ValueError
        If ``X`` has no ``release_year`` column.
    """
    if RELEASE_YEAR not in X.columns:
        raise ValueError(f"X must carry '{RELEASE_YEAR}' to be ordered in time")
    keys = [X[column].to_numpy() for column in TIME_ORDER if column in X.columns]
    # np.lexsort sorts by the last key first.
    return np.lexsort(keys[::-1])


class TemporalBoostingRegressor(RegressorMixin, BaseEstimator):
    """Preprocessing and a gradient-boosted regressor with temporal early stopping.

    Subclasses bind a boosting library by implementing
    :meth:`_make_preprocessor`, :meth:`_make_booster`, :meth:`_fit_booster`
    and :meth:`_best_n_estimators`.

    Parameters
    ----------
    config : FeatureConfig, optional
        Feature configuration of the preprocessing pipeline.
    n_estimators : int, default 2000
        Maximum number of boosting rounds.
    learning_rate : float, default 0.05
        Shrinkage :math:`\\eta` applied to each tree.
    early_stopping_rounds : int, default 100
        Rounds without improvement of the validation RMSE after which
        boosting stops.
    validation_years : int, default 2
        Latest training release years that form the inner validation fold.
        Zero disables early stopping, and ``n_estimators`` rounds are used.
    refit : bool, default True
        Refit preprocessing and booster on all training rows with the
        selected number of rounds. When False, the model fitted without the
        inner validation fold is kept.
    random_state : int, default 0
        Seed of the booster.
    params : mapping, optional
        Keyword arguments of the booster. They override
        :attr:`default_params`.

    Attributes
    ----------
    preprocessor_ : sklearn.pipeline.Pipeline
        Fitted preprocessing pipeline.
    booster_ : estimator
        Fitted booster of the library.
    n_estimators_ : int
        Number of rounds of ``booster_``: the selected :math:`M^*` when early
        stopping ran, otherwise ``n_estimators``.
    early_stopping_years_ : tuple of int or None
        First and last release year of the inner validation fold, or None
        when early stopping did not run.
    validation_rmse_ : float
        RMSE on the modelling scale of the inner validation fold at
        :math:`M^*`, or ``NaN`` when early stopping did not run.
    """

    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        config: FeatureConfig | None = None,
        n_estimators: int = 2000,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 100,
        validation_years: int = 2,
        refit: bool = True,
        random_state: int = 0,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_years = validation_years
        self.refit = refit
        self.random_state = random_state
        self.params = params

    def _make_preprocessor(self) -> Pipeline:
        raise NotImplementedError

    def _make_booster(self, n_estimators: int, early_stopping: bool) -> Any:
        raise NotImplementedError

    def _fit_booster(
        self,
        booster: Any,
        X: Any,
        y: np.ndarray,
        eval_set: tuple[Any, np.ndarray] | None,
    ) -> None:
        raise NotImplementedError

    def _best_n_estimators(self, booster: Any) -> int:
        raise NotImplementedError

    def _booster_params(self, fixed: Mapping[str, Any]) -> dict[str, Any]:
        """Merge default, user and fixed booster arguments.

        ``fixed`` holds the arguments set by the wrapper, such as the number
        of rounds and the objective; ``params`` must not override them.
        """
        params = dict(self.params or {})
        overlap = sorted(set(params) & (set(fixed) | set(WRAPPER_PARAMS)))
        if overlap:
            raise ValueError(f"params must not set {overlap}")
        return {**self.default_params, **params, **fixed}

    def _check_params(self) -> None:
        if self.n_estimators < 1 or self.early_stopping_rounds < 1:
            raise ValueError("n_estimators and early_stopping_rounds must be >= 1")
        if self.validation_years < 0:
            raise ValueError("validation_years must be >= 0")
        if not self.learning_rate > 0:
            raise ValueError("learning_rate must be positive")

    def _inner_split(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        if self.validation_years == 0:
            return None
        groups = X[GROUP_COLUMN] if GROUP_COLUMN in X.columns else None
        try:
            return holdout_split(X[RELEASE_YEAR], self.validation_years, groups)
        except ValueError:
            warnings.warn(
                f"the training rows span too few release years for a "
                f"{self.validation_years}-year validation fold; early stopping "
                f"is skipped and {self.n_estimators} rounds are used",
                UserWarning,
                stacklevel=3,
            )
            return None

    def _fit_full(self, X: pd.DataFrame, z: np.ndarray, n_estimators: int) -> None:
        self.preprocessor_ = self._make_preprocessor()
        features = self.preprocessor_.fit_transform(X, z)
        self.booster_ = self._make_booster(n_estimators, early_stopping=False)
        self._fit_booster(self.booster_, features, z, None)

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> TemporalBoostingRegressor:
        """Select the number of rounds on the latest years and fit the model.

        Parameters
        ----------
        X : pandas.DataFrame
            Pre-release feature frame with the ``release_year`` column and,
            for purging, the ``title_key`` column.
        y : array-like of float
            Target on the modelling scale.

        Returns
        -------
        TemporalBoostingRegressor
            The fitted estimator.
        """
        if not isinstance(X, pd.DataFrame) or RELEASE_YEAR not in X.columns:
            raise ValueError(f"X must be a DataFrame with a '{RELEASE_YEAR}' column")
        self._check_params()
        z = np.asarray(y, dtype=np.float64)
        if z.shape != (len(X),) or not np.isfinite(z).all():
            raise ValueError(f"y must hold {len(X)} finite values")

        inner = self._inner_split(X)
        if inner is None:
            self.early_stopping_years_ = None
            self.validation_rmse_ = float("nan")
            self.n_estimators_ = int(self.n_estimators)
            self._fit_full(X, z, self.n_estimators_)
        else:
            train_idx, val_idx = inner
            preprocessor = self._make_preprocessor()
            X_train = preprocessor.fit_transform(X.iloc[train_idx], z[train_idx])
            X_val = preprocessor.transform(X.iloc[val_idx])
            booster = self._make_booster(self.n_estimators, early_stopping=True)
            self._fit_booster(booster, X_train, z[train_idx], (X_val, z[val_idx]))
            years = X[RELEASE_YEAR].to_numpy()[val_idx]
            self.early_stopping_years_ = (int(years.min()), int(years.max()))
            self.n_estimators_ = self._best_n_estimators(booster)
            self.validation_rmse_ = _rmse(
                z[val_idx], np.asarray(booster.predict(X_val), dtype=np.float64)
            )
            if self.refit:
                self._fit_full(X, z, self.n_estimators_)
            else:
                self.preprocessor_, self.booster_ = preprocessor, booster

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target on the modelling scale."""
        features = self.preprocessor_.transform(X)
        return np.asarray(self.booster_.predict(features), dtype=np.float64)


class XGBoostRegressor(TemporalBoostingRegressor):
    """XGBoost (Chen and Guestrin, 2016) on the shared preprocessing pipeline.

    The input matrix is the output of
    :func:`vgsales.features.pipeline.make_preprocessor` without scaling:
    one-hot console and genre indicators, frequency and temporal target
    encodings of publisher and developer, and the imputed numeric features.
    Trees are grown with the histogram method on the squared error of the
    modelling-scale target.
    """

    default_params: ClassVar[dict[str, Any]] = {
        "max_depth": 6,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
    }

    def _make_preprocessor(self) -> Pipeline:
        return make_preprocessor(self.config)

    def _make_booster(self, n_estimators: int, early_stopping: bool) -> Any:
        xgb = _require("xgboost")
        fixed = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": n_estimators,
            "learning_rate": self.learning_rate,
            "early_stopping_rounds": (
                self.early_stopping_rounds if early_stopping else None
            ),
            "random_state": self.random_state,
        }
        return xgb.XGBRegressor(**self._booster_params(fixed))

    def _fit_booster(
        self,
        booster: Any,
        X: Any,
        y: np.ndarray,
        eval_set: tuple[Any, np.ndarray] | None,
    ) -> None:
        if eval_set is None:
            booster.fit(X, y, verbose=False)
        else:
            booster.fit(X, y, eval_set=[eval_set], verbose=False)

    def _best_n_estimators(self, booster: Any) -> int:
        return int(booster.best_iteration) + 1


class LightGBMRegressor(TemporalBoostingRegressor):
    """LightGBM (Ke et al., 2017) with native categorical splits.

    Console and genre enter as integer codes from
    :func:`vgsales.features.pipeline.make_preprocessor` with
    ``categorical="ordinal"`` and are declared categorical. At each node
    LightGBM orders the categories by their gradient statistics and searches
    the best binary partition along that order (Fisher, 1958), instead of
    splitting one-hot indicators one category at a time. Categories pooled as
    rare share one code, and categories unseen in training (code -1) follow
    the missing-value branch. Publisher and developer keep their frequency
    and temporal target encodings. Missing numeric values and target
    encodings are not imputed: each split learns the branch that missing
    values follow.
    """

    default_params: ClassVar[dict[str, Any]] = {
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_data_per_group": 50,
        "cat_smooth": 10.0,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }

    def _make_preprocessor(self) -> Pipeline:
        return make_preprocessor(self.config, categorical="ordinal", impute=False)

    def _categorical_features(self) -> list[int]:
        return list(range(len((self.config or FeatureConfig()).categorical)))

    def _objective(self) -> dict[str, Any]:
        """Training objective and early-stopping metric of the booster."""
        return {"objective": "regression", "metric": "rmse"}

    def _make_booster(self, n_estimators: int, early_stopping: bool) -> Any:
        lgb = _require("lightgbm")
        fixed = {
            **self._objective(),
            "n_estimators": n_estimators,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }
        return lgb.LGBMRegressor(**self._booster_params(fixed))

    def _fit_booster(
        self,
        booster: Any,
        X: Any,
        y: np.ndarray,
        eval_set: tuple[Any, np.ndarray] | None,
    ) -> None:
        kwargs: dict[str, Any] = {"categorical_feature": self._categorical_features()}
        if eval_set is not None:
            lgb = _require("lightgbm")
            # LightGBM 4.7 replaced eval_set by eval_X and eval_y.
            if "eval_X" in inspect.signature(booster.fit).parameters:
                kwargs["eval_X"], kwargs["eval_y"] = eval_set
            else:
                kwargs["eval_set"] = [eval_set]
            kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            ]
        booster.fit(X, y, **kwargs)

    def _best_n_estimators(self, booster: Any) -> int:
        return int(booster.best_iteration_)


class CatBoostRegressor(TemporalBoostingRegressor):
    r"""CatBoost (Prokhorenkova et al., 2018) with ordered target statistics.

    Console, genre, publisher and developer enter as strings from
    :func:`vgsales.features.pipeline.make_native_preprocessor`, and CatBoost
    encodes them itself. For a training row :math:`i` with category
    :math:`x_i = c` at position :math:`\sigma(i)` of a permutation
    :math:`\sigma`, the ordered target statistic is

    .. math::

        \hat x_i = \frac{\sum_{j: \sigma(j) < \sigma(i)} [x_j = c]\, z_j
                         + a\, p}
                        {\sum_{j: \sigma(j) < \sigma(i)} [x_j = c] + a},

    with prior :math:`p` and prior weight :math:`a`, so the encoding of a row
    never uses its own target. CatBoost draws :math:`\sigma` at random by
    default. Here the training rows are sorted by release year and month and
    fitted with ``has_time=True``, which makes :math:`\sigma` the release
    order: a title is encoded from titles released before it, or earlier in
    the input order within the same month. This is the in-model
    counterpart of :class:`vgsales.features.encoders.TemporalTargetEncoder`.
    Missing numeric values are handled natively. ``release_year`` must be
    among the configured numeric features.
    """

    default_params: ClassVar[dict[str, Any]] = {
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "thread_count": -1,
    }

    def _make_preprocessor(self) -> Pipeline:
        return make_native_preprocessor(self.config)

    def _make_booster(self, n_estimators: int, early_stopping: bool) -> Any:
        catboost = _require("catboost")
        config = self.config or FeatureConfig()
        fixed = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": n_estimators,
            "learning_rate": self.learning_rate,
            "random_seed": self.random_state,
            "cat_features": [*config.categorical, *config.high_cardinality],
            "has_time": True,
            "allow_writing_files": False,
            "verbose": False,
        }
        if early_stopping:
            fixed["early_stopping_rounds"] = self.early_stopping_rounds
            fixed["use_best_model"] = True
        return catboost.CatBoostRegressor(**self._booster_params(fixed))

    def _fit_booster(
        self,
        booster: Any,
        X: Any,
        y: np.ndarray,
        eval_set: tuple[Any, np.ndarray] | None,
    ) -> None:
        order = chronological_order(X)
        kwargs = {} if eval_set is None else {"eval_set": eval_set}
        booster.fit(X.iloc[order], y[order], **kwargs)

    def _best_n_estimators(self, booster: Any) -> int:
        return int(booster.get_best_iteration()) + 1


_MODELS: Final[dict[str, type[TemporalBoostingRegressor]]] = {
    "xgboost": XGBoostRegressor,
    "lightgbm": LightGBMRegressor,
    "catboost": CatBoostRegressor,
}


def make_boosting(
    name: str, config: FeatureConfig | None = None, **params: Any
) -> TemporalBoostingRegressor:
    """Build an unfitted gradient-boosting model.

    Parameters
    ----------
    name : {"xgboost", "lightgbm", "catboost"}
        Boosting library.
    config : FeatureConfig, optional
        Feature configuration of the preprocessing pipeline.
    **params
        Arguments of :class:`TemporalBoostingRegressor` listed in
        :data:`WRAPPER_PARAMS`; every other keyword is passed to the booster.

    Returns
    -------
    TemporalBoostingRegressor
        Unfitted estimator that takes the pre-release feature frame and the
        modelling-scale target.

    Raises
    ------
    ValueError
        If ``name`` is unknown.
    """
    if name not in _MODELS:
        raise ValueError(
            f"unknown boosting model {name!r}; choose from {BOOSTING_NAMES}"
        )
    wrapper = {key: params.pop(key) for key in WRAPPER_PARAMS if key in params}
    return _MODELS[name](config=config, params=params or None, **wrapper)
