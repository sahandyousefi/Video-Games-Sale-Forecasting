"""Preprocessing pipeline from pre-release features to a numeric matrix.

The pipeline expects the output of
:func:`vgsales.features.prerelease.build_prerelease_features` and the target
on the modelling scale. Every learned statistic (one-hot vocabularies,
category frequencies, temporal target means and imputation medians) is
fitted inside :meth:`fit`, so under cross-validation it is refitted on the
training rows of each fold. Transformation is row-wise: the encoding of a row
depends on the fitted statistics and on the row itself, never on the other
rows of the batch.

The first step rejects any input frame that carries the target or a column
in :data:`vgsales.data.schema.LEAKAGE_COLUMNS`, so the raw or cleaned frame
cannot be passed in place of the feature frame by mistake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from vgsales.config import FeatureConfig
from vgsales.data.schema import assert_no_leakage
from vgsales.data.temporal import RELEASE_YEAR
from vgsales.features.encoders import FrequencyEncoder, TemporalTargetEncoder

_MISSING = "__missing__"


class LeakageGuard(TransformerMixin, BaseEstimator):
    """Pass a frame through unchanged after checking it for leakage columns.

    Raises ``ValueError`` in :meth:`fit` and :meth:`transform` when the frame
    contains the target or a column of
    :data:`vgsales.data.schema.LEAKAGE_COLUMNS`.
    """

    def _check(self, X: object) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LeakageGuard expects a pandas DataFrame")
        assert_no_leakage(X.columns)
        return X

    def fit(self, X: pd.DataFrame, y: object = None) -> LeakageGuard:
        """Check the training frame and record its columns."""
        X = self._check(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check ``X`` and return it unchanged."""
        return self._check(X)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the columns seen during :meth:`fit`."""
        return self.feature_names_in_


def _fill_missing_category(X: pd.DataFrame) -> pd.DataFrame:
    out = X.astype(object)
    return out.where(out.notna(), _MISSING)


def _one_hot(min_count: int) -> Pipeline:
    return make_pipeline(
        FunctionTransformer(_fill_missing_category, feature_names_out="one-to-one"),
        OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_count,
            sparse_output=False,
        ),
    )


def _median_imputer() -> SimpleImputer:
    # Keeps columns that are entirely missing in a small training fold, so the
    # output width does not depend on the fold.
    return SimpleImputer(strategy="median", keep_empty_features=True)


def make_preprocessor(
    config: FeatureConfig | None = None, *, scale: bool = False
) -> Pipeline:
    """Build the preprocessing pipeline for the pre-release feature frame.

    Parameters
    ----------
    config : FeatureConfig, optional
        Feature section of the configuration. Defaults to ``FeatureConfig()``.
    scale : bool, default False
        Append a ``StandardScaler``, for linear and neural models.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps ``guard`` (:class:`LeakageGuard`), ``columns`` (a
        ``ColumnTransformer`` with blocks ``onehot``, ``frequency``,
        ``target`` and ``numeric``) and optionally ``scale``. Columns of the
        input that are not configured, such as the ``title_key`` group label,
        are dropped.
    """
    config = config or FeatureConfig()
    high = list(config.high_cardinality)
    blocks: list[tuple[str, object, list[str]]] = []
    if config.categorical:
        blocks.append(
            (
                "onehot",
                _one_hot(config.min_category_count),
                list(config.categorical),
            )
        )
    if high:
        blocks.append(("frequency", FrequencyEncoder(columns=high), high))
        target = make_pipeline(
            TemporalTargetEncoder.from_config(
                config.target_encoding_smoothing, columns=high
            ),
            _median_imputer(),
        )
        blocks.append(("target", target, [*high, RELEASE_YEAR]))
    if config.numeric:
        blocks.append(("numeric", _median_imputer(), list(config.numeric)))
    if not blocks:
        raise ValueError("the feature configuration selects no columns")

    steps: list[tuple[str, object]] = [
        ("guard", LeakageGuard()),
        (
            "columns",
            ColumnTransformer(
                blocks, remainder="drop", verbose_feature_names_out=False
            ),
        ),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)
