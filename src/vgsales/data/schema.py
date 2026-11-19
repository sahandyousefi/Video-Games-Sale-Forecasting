"""Column definitions and checks for the VGChartz 2024 extract.

The raw file has one row per (title, console) pair. ``total_sales`` is the
forecasting target in millions of units. The four regional sales columns are
its components and are excluded from model inputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import pandas as pd

TARGET: Final = "total_sales"
REGIONAL_SALES: Final = ("na_sales", "jp_sales", "pal_sales", "other_sales")
SALES_COLUMNS: Final = (TARGET, *REGIONAL_SALES)

IDENTIFIERS: Final = ("img", "title")
CATEGORICAL: Final = ("console", "genre", "publisher", "developer")
NUMERIC: Final = ("critic_score", *SALES_COLUMNS)
DATES: Final = ("release_date", "last_update")

RAW_COLUMNS: Final = (
    "img",
    "title",
    "console",
    "genre",
    "publisher",
    "developer",
    "critic_score",
    "total_sales",
    "na_sales",
    "jp_sales",
    "pal_sales",
    "other_sales",
    "release_date",
    "last_update",
)

# Unknown at release time: regional sales are components of the target and
# last_update records when sales tracking last changed.
LEAKAGE_COLUMNS: Final = (*REGIONAL_SALES, "last_update")

# Closed intervals; None leaves a side unbounded.
VALUE_RANGES: Final[dict[str, tuple[float | None, float | None]]] = {
    "critic_score": (0.0, 10.0),
    **{col: (0.0, None) for col in SALES_COLUMNS},
}


@dataclass(frozen=True)
class ColumnReport:
    """Result of comparing a frame's columns with the raw schema."""

    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when every schema column is present."""
        return not self.missing


def check_columns(df: pd.DataFrame) -> ColumnReport:
    """Compare the columns of ``df`` with :data:`RAW_COLUMNS`.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to check.

    Returns
    -------
    ColumnReport
        Missing schema columns and columns not in the schema.
    """
    present = set(df.columns)
    missing = tuple(c for c in RAW_COLUMNS if c not in present)
    unexpected = tuple(c for c in df.columns if c not in RAW_COLUMNS)
    return ColumnReport(missing=missing, unexpected=unexpected)


def check_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """Return columns whose dtype kind disagrees with the schema.

    String columns may use ``object`` or a pandas string dtype. Numeric columns
    must be floating or integer. Date columns must be datetime64 of any
    resolution. Columns absent from ``df`` are ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to check.

    Returns
    -------
    dict of str to str
        Mapping from column name to its observed dtype, for mismatches only.
    """
    ptypes = pd.api.types
    problems: dict[str, str] = {}
    for col in df.columns:
        dtype = df[col].dtype
        if col in IDENTIFIERS or col in CATEGORICAL:
            # Accepts object dtype as well as the pandas string dtypes.
            ok = ptypes.is_string_dtype(dtype)
        elif col in NUMERIC:
            ok = ptypes.is_numeric_dtype(dtype) and not ptypes.is_bool_dtype(dtype)
        elif col in DATES:
            ok = ptypes.is_datetime64_any_dtype(dtype)
        else:
            continue
        if not ok:
            problems[col] = str(dtype)
    return problems


def check_ranges(df: pd.DataFrame) -> dict[str, int]:
    """Count non-missing values outside :data:`VALUE_RANGES`.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with numeric columns.

    Returns
    -------
    dict of str to int
        Number of violations per column, for columns with at least one.
    """
    violations: dict[str, int] = {}
    for col, (low, high) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        values = df[col].dropna()
        bad = pd.Series(False, index=values.index)
        if low is not None:
            bad |= values < low
        if high is not None:
            bad |= values > high
        if int(bad.sum()):
            violations[col] = int(bad.sum())
    return violations


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise if any model input is a leakage column or the target itself.

    Parameters
    ----------
    columns : iterable of str
        Names of the model input columns.

    Raises
    ------
    ValueError
        If ``columns`` contains the target or a column in
        :data:`LEAKAGE_COLUMNS`.
    """
    forbidden = {TARGET, *LEAKAGE_COLUMNS}
    found = sorted(forbidden.intersection(columns))
    if found:
        raise ValueError(f"leakage columns among model inputs: {found}")
