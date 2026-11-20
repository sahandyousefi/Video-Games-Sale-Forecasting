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

When group labels :math:`G_i` (title keys) are given, a title can still span
both sides of a time boundary if it was ported to a new platform in a later
year. Such rows are purged: a training row is dropped when its group has any
row in the evaluation block. The evaluation block itself is unchanged, so
both the temporal ordering and the group separation hold.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vgsales.config import SplitConfig
from vgsales.data.temporal import RELEASE_YEAR

ArrayLike = pd.Series | np.ndarray


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
    n_purged : int
        Rows that precede the validation block but were removed from training
        because their group has rows in the block.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    train_years: tuple[int, int]
    val_years: tuple[int, int]
    n_purged: int = 0


def _as_years(years: ArrayLike) -> np.ndarray:
    values = np.asarray(years)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("years must be a non-empty one-dimensional array")
    if pd.isna(values).any():
        raise ValueError("years must not contain missing values")
    return values.astype(np.int64)


def _as_groups(groups: ArrayLike | None, n: int) -> pd.Series | None:
    if groups is None:
        return None
    labels = pd.Series(np.asarray(groups, dtype=object))
    if len(labels) != n:
        raise ValueError(f"groups has length {len(labels)}, expected {n}")
    if labels.isna().any():
        raise ValueError("groups must not contain missing values")
    return labels


def _purge(
    candidate: np.ndarray, held_out: np.ndarray, groups: pd.Series | None
) -> np.ndarray:
    """Remove from ``candidate`` the rows whose group appears in ``held_out``."""
    if groups is None:
        return candidate
    shared = groups.isin(groups[held_out].unique()).to_numpy()
    return candidate & ~shared


def holdout_split(
    years: ArrayLike, holdout_years: int, groups: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into a development set and a test set of the latest years.

    Parameters
    ----------
    years : array-like of int
        Release year of each row.
    holdout_years : int
        Number of calendar years, counted back from the latest release year,
        that form the test set.
    groups : array-like, optional
        Group label of each row, typically from
        :func:`vgsales.data.grouping.title_key`. Development rows whose group
        also has test rows are dropped from the development set.

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
    labels = _as_groups(groups, values.size)
    is_test = values > values.max() - holdout_years
    is_dev = _purge(~is_test, is_test, labels)
    if not is_dev.any():
        raise ValueError(
            f"holdout of {holdout_years} years leaves no development rows "
            f"(years {values.min()} to {values.max()})"
        )
    return np.flatnonzero(is_dev), np.flatnonzero(is_test)


def expanding_window_folds(
    years: ArrayLike,
    n_splits: int,
    min_train_years: int = 1,
    gap_years: int = 0,
    val_years: int = 1,
    groups: ArrayLike | None = None,
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
    groups : array-like, optional
        Group label of each row. Training rows whose group has rows in the
        validation block are purged.

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
    labels = _as_groups(groups, values.size)
    last = int(values.max())
    folds: list[Fold] = []
    for k in range(n_splits):
        val_end = last - (n_splits - 1 - k) * val_years
        val_start = val_end - val_years + 1
        in_val = (values >= val_start) & (values <= val_end)
        if not in_val.any():
            raise ValueError(f"validation block {val_start}-{val_end} has no rows")
        before = values < val_start - gap_years
        in_train = _purge(before, in_val, labels)
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
                n_purged=int(before.sum() - in_train.sum()),
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
        ``train_end``, ``val_start``, ``val_end``, ``n_train``, ``n_val`` and
        ``n_purged``.
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
                "n_purged": f.n_purged,
            }
            for k, f in enumerate(folds)
        ]
    )


class ExpandingWindowSplit:
    """Scikit-learn compatible expanding-window splitter over release years.

    The release year is read from column ``time_column`` of ``X``, so ``X``
    must be a DataFrame that carries it. Group labels are taken from the
    ``groups`` argument of :meth:`split`, which scikit-learn model selection
    utilities forward, or else from ``group_column`` of ``X`` when set.

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
    group_column : str, optional
        Column of ``X`` holding group labels, used when ``groups`` is not
        passed to :meth:`split`.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_years: int = 10,
        gap_years: int = 0,
        val_years: int = 1,
        time_column: str = RELEASE_YEAR,
        group_column: str | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_years = min_train_years
        self.gap_years = gap_years
        self.val_years = val_years
        self.time_column = time_column
        self.group_column = group_column

    @classmethod
    def from_config(
        cls,
        config: SplitConfig,
        val_years: int = 1,
        time_column: str = RELEASE_YEAR,
        group_column: str | None = None,
    ) -> ExpandingWindowSplit:
        """Create a splitter from the ``split`` configuration section."""
        return cls(
            n_splits=config.cv_folds,
            min_train_years=config.min_train_years,
            gap_years=config.gap_years,
            val_years=val_years,
            time_column=time_column,
            group_column=group_column,
        )

    def _column(self, X: pd.DataFrame, name: str) -> pd.Series:
        if not isinstance(X, pd.DataFrame) or name not in X.columns:
            raise ValueError(f"X must be a DataFrame with a '{name}' column")
        return X[name]

    def folds(self, X: pd.DataFrame, groups: ArrayLike | None = None) -> list[Fold]:
        """Return the folds for ``X`` with their year ranges."""
        if groups is None and self.group_column is not None:
            groups = self._column(X, self.group_column)
        return expanding_window_folds(
            self._column(X, self.time_column),
            n_splits=self.n_splits,
            min_train_years=self.min_train_years,
            gap_years=self.gap_years,
            val_years=self.val_years,
            groups=groups,
        )

    def split(
        self,
        X: pd.DataFrame,
        y: object = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield positional train and validation indices for each fold."""
        for fold in self.folds(X, groups):
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
            f"val_years={self.val_years}, time_column={self.time_column!r}, "
            f"group_column={self.group_column!r})"
        )
