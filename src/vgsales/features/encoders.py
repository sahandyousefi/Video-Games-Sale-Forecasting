"""Categorical encoders whose statistics come from training rows only.

Each encoder is a scikit-learn transformer, so inside a ``Pipeline`` its
statistics are refitted on the training part of every cross-validation fold
and never see validation or test rows (known issue 5 in ``ROADMAP.md``).
:class:`TemporalTargetEncoder` additionally restricts the target statistics
of each row to strictly earlier release years, which replaces the full-data
group means of the original notebook (known issue 4).
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

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


class TemporalTargetEncoder(TransformerMixin, BaseEstimator):
    """Smoothed target mean of each category over strictly earlier years.

    Let :math:`t_i` be the release year of row :math:`i`, :math:`c_i` its
    category in column :math:`j` and :math:`y` the target on the modelling
    scale. With :math:`\\mathcal{H}(t) = \\{k : t_k < t\\}` the training rows
    released before year :math:`t`, define

    .. math::

        n_c(t) = \\sum_{k \\in \\mathcal{H}(t)} [c_k = c], \\quad
        S_c(t) = \\sum_{k \\in \\mathcal{H}(t)} [c_k = c]\\, y_k, \\quad
        \\mu(t) = \\frac{1}{|\\mathcal{H}(t)|}
                   \\sum_{k \\in \\mathcal{H}(t)} y_k .

    The encoding of a row with category :math:`c` and year :math:`t` is the
    shrinkage estimate

    .. math::

        \\hat m_c(t) = \\frac{S_c(t) + \\lambda\\, \\mu(t)}{n_c(t) + \\lambda},

    where :math:`\\lambda` is ``smoothing``. Categories with no earlier rows
    receive the prior :math:`\\mu(t)`. Rows of the same year never inform
    each other, so ``fit_transform`` on the training rows yields out-of-time
    encodings, and adding rows of later years to the training data leaves
    the encodings of earlier rows unchanged. Rows whose year has no earlier
    training data are encoded as ``NaN``; in a pipeline these occur only in
    the earliest training years and must be imputed or handled natively by
    the model. Missing categories form their own category.

    Parameters
    ----------
    columns : sequence of str, default ("publisher", "developer")
        Categorical columns to encode.
    time_column : str, default "release_year"
        Column holding the integer release year. It is read but not encoded.
    smoothing : float, default 20.0
        Prior weight :math:`\\lambda`, in pseudo-rows. Zero gives the raw
        mean of earlier rows.

    Attributes
    ----------
    years_ : numpy.ndarray
        Sorted distinct training years.
    prior_ : numpy.ndarray
        :math:`\\mu` before each training year and after the last one, of
        length ``len(years_) + 1``. The first entry is ``NaN``.
    categories_ : dict of str to pandas.Index
        Training categories per column.
    sums_, counts_ : dict of str to numpy.ndarray
        Cumulative :math:`S_c` and :math:`n_c` per column, of shape
        ``(n_categories, len(years_) + 1)``; column :math:`k` aggregates the
        first :math:`k` training years.
    feature_names_in_ : numpy.ndarray
        Encoded columns followed by ``time_column``.
    n_features_in_ : int
        Number of columns read during :meth:`fit`.
    """

    def __init__(
        self,
        columns: Sequence[str] = ("publisher", "developer"),
        time_column: str = "release_year",
        smoothing: float = 20.0,
    ) -> None:
        self.columns = columns
        self.time_column = time_column
        self.smoothing = smoothing

    @classmethod
    def from_config(
        cls,
        smoothing: float,
        columns: Sequence[str] = ("publisher", "developer"),
        time_column: str = "release_year",
    ) -> TemporalTargetEncoder:
        """Create an encoder with ``features.target_encoding_smoothing``."""
        return cls(columns=columns, time_column=time_column, smoothing=smoothing)

    def _check_frame(self, X: object, columns: Sequence[str]) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TemporalTargetEncoder expects a pandas DataFrame")
        missing = sorted(set(columns) - set(X.columns))
        if missing:
            raise ValueError(f"columns not found in X: {missing}")
        years = X[self.time_column]
        if years.isna().any():
            raise ValueError(f"{self.time_column} must not contain missing values")
        return X

    def _year_position(self, years: pd.Series) -> np.ndarray:
        # Number of training years strictly before each row's year.
        return np.searchsorted(self.years_, years.to_numpy(np.int64), side="left")

    def fit(self, X: pd.DataFrame, y: object = None) -> TemporalTargetEncoder:
        """Accumulate per-year target sums and counts for each category.

        Parameters
        ----------
        X : pandas.DataFrame
            Training rows with the encoded columns and ``time_column``.
        y : array-like of shape (n_samples,)
            Target on the modelling scale, for example ``log1p`` of sales.

        Returns
        -------
        TemporalTargetEncoder
            The fitted encoder.
        """
        if not isinstance(self.smoothing, Real) or self.smoothing < 0:
            raise ValueError("smoothing must be a non-negative number")
        if y is None:
            raise ValueError("TemporalTargetEncoder requires y")
        columns = list(self.columns)
        X = self._check_frame(X, [*columns, self.time_column])
        target = np.asarray(y, dtype="float64")
        if target.shape != (len(X),):
            raise ValueError(f"y must have shape ({len(X)},), got {target.shape}")
        if len(X) == 0:
            raise ValueError("cannot fit on an empty frame")
        if not np.isfinite(target).all():
            raise ValueError("y must be finite")

        self.years_, year_idx = np.unique(
            X[self.time_column].to_numpy(np.int64), return_inverse=True
        )
        n_years = self.years_.size
        totals = np.bincount(year_idx, weights=target, minlength=n_years)
        sizes = np.bincount(year_idx, minlength=n_years).astype("float64")
        cum_totals = np.concatenate([[0.0], np.cumsum(totals)])
        cum_sizes = np.concatenate([[0.0], np.cumsum(sizes)])
        with np.errstate(invalid="ignore", divide="ignore"):
            self.prior_ = np.where(cum_sizes > 0, cum_totals / cum_sizes, np.nan)

        self.categories_: dict[str, pd.Index] = {}
        self.sums_: dict[str, np.ndarray] = {}
        self.counts_: dict[str, np.ndarray] = {}
        for col in columns:
            codes, uniques = pd.factorize(_as_categories(X[col]))
            n_cat = uniques.size
            flat = codes * n_years + year_idx
            size = n_cat * n_years
            sums = np.bincount(flat, weights=target, minlength=size)
            counts = np.bincount(flat, minlength=size).astype("float64")
            shape = (n_cat, n_years)
            zeros = np.zeros((n_cat, 1))
            self.categories_[col] = pd.Index(uniques, dtype=object)
            self.sums_[col] = np.hstack([zeros, sums.reshape(shape).cumsum(axis=1)])
            self.counts_[col] = np.hstack([zeros, counts.reshape(shape).cumsum(axis=1)])
        self.feature_names_in_ = np.asarray([*columns, self.time_column], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Encode each row from training rows of strictly earlier years.

        Parameters
        ----------
        X : pandas.DataFrame
            Rows with the encoded columns and ``time_column``.

        Returns
        -------
        numpy.ndarray
            Float array of shape ``(len(X), len(columns))``.
        """
        check_is_fitted(self, "prior_")
        columns = list(self.feature_names_in_[:-1])
        X = self._check_frame(X, [*columns, self.time_column])
        position = self._year_position(X[self.time_column])
        prior = self.prior_[position]
        lam = float(self.smoothing)
        encoded = []
        for col in columns:
            codes = self.categories_[col].get_indexer(_as_categories(X[col]))
            seen = codes >= 0
            sums = np.zeros(len(X))
            counts = np.zeros(len(X))
            sums[seen] = self.sums_[col][codes[seen], position[seen]]
            counts[seen] = self.counts_[col][codes[seen], position[seen]]
            denominator = counts + lam
            with np.errstate(invalid="ignore", divide="ignore"):
                value = (sums + lam * prior) / denominator
            encoded.append(np.where(denominator > 0, value, prior))
        return np.column_stack(encoded) if encoded else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names of the form ``<column>_te``."""
        check_is_fitted(self, "prior_")
        return np.asarray(
            [f"{col}_te" for col in self.feature_names_in_[:-1]], dtype=object
        )
