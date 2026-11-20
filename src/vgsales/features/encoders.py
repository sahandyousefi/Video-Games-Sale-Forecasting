"""Categorical encoders whose statistics come from training rows only.

Each encoder is a scikit-learn transformer, so inside a ``Pipeline`` its
statistics are refitted on the training part of every cross-validation fold
and never see validation or test rows (known issue 5 in ``ROADMAP.md``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Missing values are encoded as their own category.
_MISSING = "__missing__"


def _as_categories(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), _MISSING)


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category by its frequency in the training data.

    For column :math:`j` and category :math:`c`, the fitted value is
    :math:`\\hat p_j(c) = n_j(c) / n`, where :math:`n_j(c)` is the number of
    training rows with that category and :math:`n` the number of training
    rows. Using proportions rather than counts keeps the scale comparable
    across expanding training windows of different size. Missing values form
    their own category. Categories absent from the training data map to
    ``unseen_value``.

    Parameters
    ----------
    columns : sequence of str, optional
        Columns to encode. Defaults to every column of the frame passed to
        :meth:`fit`.
    normalize : bool, default True
        If False, encode raw counts instead of proportions.
    unseen_value : float, default 0.0
        Value for categories not observed during fitting.

    Attributes
    ----------
    frequencies_ : dict of str to pandas.Series
        Fitted value per category, for each column.
    feature_names_in_ : numpy.ndarray
        Columns seen during :meth:`fit`.
    n_features_in_ : int
        Number of columns seen during :meth:`fit`.
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        normalize: bool = True,
        unseen_value: float = 0.0,
    ) -> None:
        self.columns = columns
        self.normalize = normalize
        self.unseen_value = unseen_value

    def _check_frame(self, X: object) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FrequencyEncoder expects a pandas DataFrame")
        return X

    def fit(self, X: pd.DataFrame, y: object = None) -> FrequencyEncoder:
        """Compute category frequencies from the training rows.

        Parameters
        ----------
        X : pandas.DataFrame
            Training rows.
        y : ignored

        Returns
        -------
        FrequencyEncoder
            The fitted encoder.
        """
        X = self._check_frame(X)
        columns = list(X.columns) if self.columns is None else list(self.columns)
        missing = sorted(set(columns) - set(X.columns))
        if missing:
            raise ValueError(f"columns not found in X: {missing}")
        if len(X) == 0:
            raise ValueError("cannot fit on an empty frame")
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.n_features_in_ = len(columns)
        self.frequencies_ = {
            col: _as_categories(X[col]).value_counts(normalize=self.normalize)
            for col in columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Map categories to their training frequencies.

        Parameters
        ----------
        X : pandas.DataFrame
            Rows to encode. Must contain the fitted columns.

        Returns
        -------
        numpy.ndarray
            Float array of shape ``(len(X), n_features_in_)``.
        """
        check_is_fitted(self, "frequencies_")
        X = self._check_frame(X)
        columns = list(self.feature_names_in_)
        missing = sorted(set(columns) - set(X.columns))
        if missing:
            raise ValueError(f"columns not found in X: {missing}")
        encoded = [
            _as_categories(X[col])
            .map(self.frequencies_[col])
            .astype("float64")
            .fillna(self.unseen_value)
            .to_numpy()
            for col in columns
        ]
        return np.column_stack(encoded) if encoded else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names of the form ``<column>_freq``."""
        check_is_fitted(self, "frequencies_")
        return np.asarray(
            [f"{col}_freq" for col in self.feature_names_in_], dtype=object
        )
