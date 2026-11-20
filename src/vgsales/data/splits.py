"""Release-year holdout and expanding-window cross-validation.

Let :math:`t_i` be the release year of row :math:`i` and :math:`T` the latest
release year in the frame. The holdout split assigns rows with
:math:`t_i > T - h` to the test set, where :math:`h` is
``split.holdout_years``. Cross-validation runs on the remaining rows only.

The expanding-window cross-validator builds :math:`K` folds whose validation
blocks are consecutive runs of :math:`v` calendar years ending at the latest
development year. For fold :math:`k` with validation years
:math:`[a_k, b_k]`, the training set is every row with
:math:`t_i < a_k - g`, where :math:`g` is ``split.gap_years``. Training
windows therefore grow with :math:`k`, and no fold trains on a year later
than the one it validates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vgsales.config import SplitConfig
from vgsales.data.temporal import RELEASE_YEAR


@dataclass(frozen=True)
class Fold:
    """One train and validation split with its year ranges.

    Attributes
    ----------
    train_idx, val_idx : numpy.ndarray
        Positional row indices.
    train_years : tuple of int
        First and last release year in the training set.
    val_years : tuple of int
        First and last calendar year of the validation block.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    train_years: tuple[int, int]
    val_years: tuple[int, int]


def _as_years(years: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(years)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("years must be a non-empty one-dimensional array")
    if pd.isna(values).any():
        raise ValueError("years must not contain missing values")
    return values.astype(np.int64)


def holdout_split(
    years: pd.Series | np.ndarray, holdout_years: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into a development set and a test set of the latest years.

    Parameters
    ----------
    years : array-like of int
        Release year of each row.
    holdout_years : int
        Number of calendar years, counted back from the latest release year,
        that form the test set.

    Returns
    -------
    dev_idx, test_idx : numpy.ndarray
        Positional indices of the development and test rows.

    Raises
    ------
    ValueError
        If ``holdout_years`` is not positive or the development set would be
        empty.
    """
    if holdout_years < 1:
        raise ValueError("holdout_years must be >= 1")
    values = _as_years(years)
    cutoff = values.max() - holdout_years
    is_test = values > cutoff
    if is_test.all():
        raise ValueError(
            f"holdout of {holdout_years} years leaves no development rows "
            f"(years {values.min()} to {values.max()})"
        )
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def expanding_window_folds(
    years: pd.Series | np.ndarray,
    n_splits: int,
    min_train_years: int = 1,
    gap_years: int = 0,
    val_years: int = 1,
) -> list[Fold]:
    """Build expanding-window folds over release years.

    Parameters
    ----------
    years : array-like of int
        Release year of each row.
    n_splits : int
        Number of folds.
    min_train_years : int, default 1
        Minimum number of distinct release years in the training set of the
        first fold.
    gap_years : int, default 0
        Calendar years left out between the end of training and the start of
        validation.
    val_years : int, default 1
        Calendar years in each validation block.

    Returns
    -------
    list of Fold
        Folds ordered from the earliest to the latest validation block.

    Raises
    ------
    ValueError
        If a validation block has no rows or the first training window has
        fewer than ``min_train_years`` distinct years.
    """
    if n_splits < 1 or min_train_years < 1 or val_years < 1 or gap_years < 0:
        raise ValueError(
            "n_splits, min_train_years and val_years must be >= 1 and "
            "gap_years must be >= 0"
        )
    values = _as_years(years)
    last = int(values.max())
    folds: list[Fold] = []
    for k in range(n_splits):
        val_end = last - (n_splits - 1 - k) * val_years
        val_start = val_end - val_years + 1
        in_val = (values >= val_start) & (values <= val_end)
        in_train = values < val_start - gap_years
        if not in_val.any():
            raise ValueError(f"validation block {val_start}-{val_end} has no rows")
        n_train_years = np.unique(values[in_train]).size
        if n_train_years < min_train_years:
            raise ValueError(
                f"fold {k} trains on {n_train_years} distinct years before "
                f"{val_start - gap_years}, fewer than min_train_years="
                f"{min_train_years}; reduce n_splits or min_train_years"
            )
        train_years = values[in_train]
        folds.append(
            Fold(
                train_idx=np.flatnonzero(in_train),
                val_idx=np.flatnonzero(in_val),
                train_years=(int(train_years.min()), int(train_years.max())),
                val_years=(val_start, val_end),
            )
        )
    return folds


def describe_folds(folds: list[Fold]) -> pd.DataFrame:
    """Tabulate year ranges and sizes of a list of folds.

    Parameters
    ----------
    folds : list of Fold
        Output of :func:`expanding_window_folds`.

    Returns
    -------
    pandas.DataFrame
        One row per fold with columns ``fold``, ``train_start``,
        ``train_end``, ``val_start``, ``val_end``, ``n_train`` and ``n_val``.
    """
    return pd.DataFrame(
        [
            {
                "fold": k,
                "train_start": f.train_years[0],
                "train_end": f.train_years[1],
                "val_start": f.val_years[0],
                "val_end": f.val_years[1],
                "n_train": f.train_idx.size,
                "n_val": f.val_idx.size,
            }
            for k, f in enumerate(folds)
        ]
    )


class ExpandingWindowSplit:
    """Scikit-learn compatible expanding-window splitter over release years.

    The release year is read from column ``time_column`` of ``X``, so ``X``
    must be a DataFrame that carries it. The splitter can be passed as ``cv``
    to scikit-learn model selection utilities.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    min_train_years : int, default 10
        Minimum number of distinct training years in the first fold.
    gap_years : int, default 0
        Calendar years between training and validation.
    val_years : int, default 1
        Calendar years in each validation block.
    time_column : str, default "release_year"
        Column of ``X`` holding the release year.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_years: int = 10,
        gap_years: int = 0,
        val_years: int = 1,
        time_column: str = RELEASE_YEAR,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_years = min_train_years
        self.gap_years = gap_years
        self.val_years = val_years
        self.time_column = time_column

    @classmethod
    def from_config(
        cls,
        config: SplitConfig,
        val_years: int = 1,
        time_column: str = RELEASE_YEAR,
    ) -> ExpandingWindowSplit:
        """Create a splitter from the ``split`` configuration section."""
        return cls(
            n_splits=config.cv_folds,
            min_train_years=config.min_train_years,
            gap_years=config.gap_years,
            val_years=val_years,
            time_column=time_column,
        )

    def _years(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame) or self.time_column not in X.columns:
            raise ValueError(
                f"X must be a DataFrame with a '{self.time_column}' column"
            )
        return _as_years(X[self.time_column])

    def folds(self, X: pd.DataFrame) -> list[Fold]:
        """Return the folds for ``X`` with their year ranges."""
        return expanding_window_folds(
            self._years(X),
            n_splits=self.n_splits,
            min_train_years=self.min_train_years,
            gap_years=self.gap_years,
            val_years=self.val_years,
        )

    def split(
        self, X: pd.DataFrame, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield positional train and validation indices for each fold."""
        for fold in self.folds(X):
            yield fold.train_idx, fold.val_idx

    def get_n_splits(
        self, X: object = None, y: object = None, groups: object = None
    ) -> int:
        """Return the number of folds."""
        return self.n_splits

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_splits={self.n_splits}, "
            f"min_train_years={self.min_train_years}, gap_years={self.gap_years}, "
            f"val_years={self.val_years}, time_column={self.time_column!r})"
        )
