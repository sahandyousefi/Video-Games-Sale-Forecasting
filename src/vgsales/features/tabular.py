r"""Inputs of a network with entity embeddings.

:class:`TabularEncoder` maps the pre-release feature frame to two arrays:

- an integer matrix :math:`C \in \{0, \dots, K_j - 1\}^{n \times m}` with one
  column per categorical variable :math:`j` (the ``categorical`` and
  ``high_cardinality`` columns of the configuration), whose entries index
  rows of an embedding table of size :math:`K_j`;
- a float matrix :math:`U \in \mathbb{R}^{n \times d}` of standardized
  numeric inputs: the configured numeric features and, by default, the
  frequency and temporal target encodings of the high-cardinality columns.

For column :math:`j`, the categories with at least ``min_category_count``
training rows receive the indices :math:`1, \dots, K_j - 1` in sorted order.
Index :data:`OOV_INDEX` (zero) is shared by rare categories and by categories
not seen in training, so the embedding of index zero is learned from the
rare training categories and serves as the fallback for new publishers and
developers. Missing values form their own category, ``"__missing__"``,
subject to the same frequency rule.

All statistics (vocabularies, frequencies, temporal target means, imputation
medians, means and standard deviations) are learned in :meth:`fit`, and the
encoding of a row does not depend on the other rows of the batch. The module
uses NumPy and scikit-learn only; the tensors are built in
:mod:`vgsales.models.neural.data`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from vgsales.config import FeatureConfig
from vgsales.data.schema import assert_no_leakage
from vgsales.features.pipeline import make_preprocessor

OOV_INDEX: Final = 0
MISSING_CATEGORY: Final = "__missing__"


@dataclass(frozen=True)
class TabularArrays:
    """Encoded rows: embedding indices and standardized numeric inputs.

    Attributes
    ----------
    categorical : numpy.ndarray of int64, shape (n_rows, n_categorical)
        Embedding index of each categorical column.
    numeric : numpy.ndarray of float32, shape (n_rows, n_numeric)
        Standardized numeric inputs.
    """

    categorical: np.ndarray
    numeric: np.ndarray

    def __post_init__(self) -> None:
        if self.categorical.ndim != 2 or self.numeric.ndim != 2:
            raise ValueError("categorical and numeric must be 2-dimensional")
        if self.categorical.shape[0] != self.numeric.shape[0]:
            raise ValueError(
                f"categorical has {self.categorical.shape[0]} rows and numeric "
                f"has {self.numeric.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.categorical.shape[0])

    def subset(self, index: ArrayLike) -> TabularArrays:
        """Return the rows at the positions ``index``."""
        index = np.asarray(index)
        return TabularArrays(self.categorical[index], self.numeric[index])


def _as_categories(values: pd.Series) -> pd.Series:
    out = values.astype(object)
    return out.where(out.notna(), MISSING_CATEGORY).astype(str)


class TabularEncoder(BaseEstimator):
    """Encode the pre-release feature frame for an entity-embedding network.

    Parameters
    ----------
    config : FeatureConfig, optional
        Feature section of the configuration. Defaults to ``FeatureConfig()``.
    high_cardinality_encodings : bool, default True
        Append the training frequency and the temporal target encoding of
        each high-cardinality column to the numeric inputs, as produced by
        :func:`vgsales.features.pipeline.make_preprocessor`. The target is
        then required in :meth:`fit`.

    Attributes
    ----------
    embedding_columns_ : tuple of str
        Columns encoded as embedding indices, in output order.
    vocabularies_ : list of dict
        For each embedding column, the map from category to index.
    cardinalities_ : tuple of int
        Embedding table size :math:`K_j` of each column, index zero included.
    numeric_features_ : numpy.ndarray of str
        Names of the numeric inputs, in output order.
    numeric_pipeline_ : sklearn.pipeline.Pipeline or None
        Fitted imputation, encoding and scaling of the numeric inputs, or
        None when there are none.
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        high_cardinality_encodings: bool = True,
    ) -> None:
        self.config = config
        self.high_cardinality_encodings = high_cardinality_encodings

    def _check_frame(self, X: object) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabularEncoder expects a pandas DataFrame")
        assert_no_leakage(X.columns)
        return X

    def _numeric_config(self, config: FeatureConfig) -> FeatureConfig:
        high = config.high_cardinality if self.high_cardinality_encodings else ()
        return replace(config, categorical=(), high_cardinality=high)

    def fit(self, X: pd.DataFrame, y: ArrayLike | None = None) -> TabularEncoder:
        """Learn the vocabularies and the numeric preprocessing.

        Parameters
        ----------
        X : pandas.DataFrame
            Pre-release feature frame.
        y : array-like of float, optional
            Target on the modelling scale. Required when
            ``high_cardinality_encodings`` is True and the configuration has
            high-cardinality columns.

        Returns
        -------
        TabularEncoder
            The fitted encoder.
        """
        X = self._check_frame(X)
        config = self.config or FeatureConfig()
        self.embedding_columns_ = (*config.categorical, *config.high_cardinality)
        missing = sorted(set(self.embedding_columns_) - set(X.columns))
        if missing:
            raise ValueError(f"X is missing categorical columns: {missing}")

        self.vocabularies_ = []
        for column in self.embedding_columns_:
            counts = _as_categories(X[column]).value_counts()
            kept = sorted(counts.index[counts >= config.min_category_count])
            self.vocabularies_.append(
                {category: i + 1 for i, category in enumerate(kept)}
            )
        self.cardinalities_ = tuple(len(v) + 1 for v in self.vocabularies_)

        numeric_config = self._numeric_config(config)
        if numeric_config.high_cardinality and y is None:
            raise ValueError("y is required to fit the temporal target encodings")
        self.numeric_pipeline_: Pipeline | None = None
        self.numeric_features_ = np.asarray([], dtype=object)
        if numeric_config.high_cardinality or numeric_config.numeric:
            target = None if y is None else np.asarray(y, dtype=np.float64)
            self.numeric_pipeline_ = make_preprocessor(numeric_config, scale=True)
            self.numeric_pipeline_.fit(X, target)
            self.numeric_features_ = self.numeric_pipeline_.get_feature_names_out()

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    @property
    def n_numeric_(self) -> int:
        """Number of numeric inputs."""
        return len(self.numeric_features_)

    def transform(self, X: pd.DataFrame) -> TabularArrays:
        """Encode ``X`` with the fitted vocabularies and numeric pipeline.

        Returns
        -------
        TabularArrays
            Embedding indices (int64) and standardized numeric inputs
            (float32). Missing numeric values are imputed with the training
            medians before scaling.
        """
        X = self._check_frame(X)
        codes = np.empty((len(X), len(self.embedding_columns_)), dtype=np.int64)
        for j, (column, vocabulary) in enumerate(
            zip(self.embedding_columns_, self.vocabularies_, strict=True)
        ):
            mapped = _as_categories(X[column]).map(vocabulary)
            codes[:, j] = mapped.fillna(OOV_INDEX).to_numpy(dtype=np.int64)
        if self.numeric_pipeline_ is None:
            numeric = np.zeros((len(X), 0), dtype=np.float32)
        else:
            numeric = np.asarray(self.numeric_pipeline_.transform(X), dtype=np.float32)
        return TabularArrays(codes, numeric)

    def fit_transform(
        self, X: pd.DataFrame, y: ArrayLike | None = None
    ) -> TabularArrays:
        """Fit the encoder on ``X`` and return the encoded training rows."""
        return self.fit(X, y).transform(X)
