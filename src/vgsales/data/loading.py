"""Read the raw VGChartz extract and coerce it to the schema dtypes.

The loader is deliberately conservative: it does not drop rows or impute
values. Text columns become ``object`` with ``NaN`` for missing values (the
same representation under pandas 2 and 3, and the one scikit-learn encoders
expect), numeric columns become ``float64`` and date columns become
``datetime64[ns]``. Values that cannot be parsed either raise or become
missing, depending on ``strict``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from vgsales.data.schema import (
    CATEGORICAL,
    DATES,
    IDENTIFIERS,
    NUMERIC,
    RAW_COLUMNS,
    VALUE_RANGES,
    check_columns,
    check_dtypes,
    check_ranges,
)

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when raw data does not conform to the schema."""


@dataclass(frozen=True)
class LoadReport:
    """Summary of the coercion applied to a raw frame.

    Attributes
    ----------
    n_rows : int
        Number of rows in the returned frame.
    dropped_columns : tuple of str
        Columns not in the schema, removed from the output.
    unparseable : dict of str to int
        Per column, the number of non-missing raw values that could not be
        parsed into the schema dtype and were set to missing.
    out_of_range : dict of str to int
        Per column, the number of values outside the schema range that were
        set to missing.
    """

    n_rows: int
    dropped_columns: tuple[str, ...] = ()
    unparseable: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)


def _as_text(series: pd.Series) -> pd.Series:
    out = series.astype(object)
    present = out.notna()
    out[present] = out[present].astype(str)
    return out.where(present, np.nan)


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _as_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    else:
        parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
    return parsed.dt.as_unit("ns")


def _mask_out_of_range(df: pd.DataFrame) -> dict[str, int]:
    counts = check_ranges(df)
    for col in counts:
        low, high = VALUE_RANGES[col]
        values = df[col]
        bad = pd.Series(False, index=df.index)
        if low is not None:
            bad |= values < low
        if high is not None:
            bad |= values > high
        df.loc[bad, col] = np.nan
    return counts


def coerce_raw(
    df: pd.DataFrame, *, strict: bool = True
) -> tuple[pd.DataFrame, LoadReport]:
    """Coerce a raw frame to the schema column order and dtypes.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with at least the columns of
        :data:`vgsales.data.schema.RAW_COLUMNS`, typically as returned by
        ``pandas.read_csv``.
    strict : bool, default True
        If True, unparseable values and values outside the schema ranges raise
        :class:`SchemaError`. If False, they are set to missing and counted in
        the report.

    Returns
    -------
    frame : pandas.DataFrame
        Copy of ``df`` restricted to the schema columns, in schema order.
    report : LoadReport
        Counts of dropped columns and coerced values.

    Raises
    ------
    SchemaError
        If schema columns are missing, or, when ``strict`` is True, if values
        cannot be parsed or fall outside their range.
    """
    columns = check_columns(df)
    if not columns.ok:
        raise SchemaError(f"missing columns: {list(columns.missing)}")

    out = df.loc[:, list(RAW_COLUMNS)].copy()
    converters = {
        **{col: _as_text for col in (*IDENTIFIERS, *CATEGORICAL)},
        **{col: _as_float for col in NUMERIC},
        **{col: _as_datetime for col in DATES},
    }
    unparseable: dict[str, int] = {}
    for col, convert in converters.items():
        before = out[col].notna()
        out[col] = convert(out[col])
        lost = int((before & out[col].isna()).sum())
        if lost:
            unparseable[col] = lost
    if strict and unparseable:
        raise SchemaError(f"unparseable values per column: {unparseable}")

    if strict:
        violations = check_ranges(out)
        if violations:
            raise SchemaError(f"values outside schema ranges: {violations}")
        out_of_range: dict[str, int] = {}
    else:
        out_of_range = _mask_out_of_range(out)

    # Guards against a converter regression rather than bad input.
    dtype_problems = check_dtypes(out)
    if dtype_problems:
        raise SchemaError(f"unexpected dtypes after coercion: {dtype_problems}")

    report = LoadReport(
        n_rows=len(out),
        dropped_columns=columns.unexpected,
        unparseable=unparseable,
        out_of_range=out_of_range,
    )
    return out.reset_index(drop=True), report


def load_raw(path: str | Path, *, strict: bool = True) -> pd.DataFrame:
    """Read the raw CSV and coerce it to the schema.

    All columns are read as text first so that type inference cannot
    reinterpret identifiers (for example a numeric-looking title), and are
    then converted by :func:`coerce_raw`.

    Parameters
    ----------
    path : str or Path
        Location of ``vgchartz-2024.csv`` or a file with the same layout.
    strict : bool, default True
        Passed to :func:`coerce_raw`.

    Returns
    -------
    pandas.DataFrame
        Frame with the schema columns and dtypes.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    SchemaError
        If the file does not conform to the schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; see data/README.md for acquisition instructions"
        )
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    frame, report = coerce_raw(raw, strict=strict)
    logger.info(
        "loaded %d rows from %s (dropped columns: %s, unparseable: %s, "
        "out of range: %s)",
        report.n_rows,
        path,
        list(report.dropped_columns),
        report.unparseable,
        report.out_of_range,
    )
    return frame
