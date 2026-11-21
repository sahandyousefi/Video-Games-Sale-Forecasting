"""Descriptive summaries for the exploratory analysis.

The functions describe the extract and are not used to build model inputs.
They take frames at different stages: :func:`missingness` and
:func:`coverage_by_year` are meant for the loaded frame before cleaning, where
untracked rows are still present, while :func:`target_summary` and
:func:`censoring_summary` describe observed sales after cleaning.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from vgsales.data.schema import TARGET
from vgsales.data.temporal import CENSORED, RELEASE_YEAR

QUANTILES: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.99)


def _require(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    if len(frame) == 0:
        raise ValueError("frame has no rows")


def missingness(
    frame: pd.DataFrame, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Count missing values per column.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame to describe.
    columns : sequence of str, optional
        Columns to include. Defaults to all columns.

    Returns
    -------
    pandas.DataFrame
        Index ``column``; columns ``n_missing`` and ``share_missing``, sorted
        by decreasing share.
    """
    columns = list(frame.columns) if columns is None else list(columns)
    _require(frame, columns)
    n_missing = frame[columns].isna().sum().astype("int64")
    share = n_missing / len(frame)
    out = pd.DataFrame({"n_missing": n_missing, "share_missing": share})
    out.index.name = "column"
    return out.sort_values("share_missing", ascending=False, kind="stable")


def coverage_by_year(frame: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Tabulate rows, tracked rows and censoring by release year.

    Parameters
    ----------
    frame : pandas.DataFrame
        Release-scoped frame with ``release_year`` and the target. It should
        still contain untracked rows, so that the tracked share is
        meaningful. A ``censored`` column, if present, is counted.
    target : str, default "total_sales"
        Target column; a missing value means the title was not tracked.

    Returns
    -------
    pandas.DataFrame
        Index ``release_year``; columns ``n_rows``, ``n_tracked``,
        ``tracked_share``, ``median_sales`` (over tracked rows) and, when
        available, ``n_censored``.
    """
    _require(frame, [RELEASE_YEAR, target])
    sales = frame[target]
    grouped = sales.groupby(frame[RELEASE_YEAR])
    out = pd.DataFrame(
        {
            "n_rows": grouped.size().astype("int64"),
            "n_tracked": grouped.count().astype("int64"),
            "median_sales": grouped.median(),
        }
    )
    out.insert(2, "tracked_share", out["n_tracked"] / out["n_rows"])
    if CENSORED in frame.columns:
        censored = frame[CENSORED].astype(bool).groupby(frame[RELEASE_YEAR]).sum()
        out["n_censored"] = censored.astype("int64")
    out.index.name = RELEASE_YEAR
    return out


def top_share(sales: pd.Series, fraction: float) -> float:
    """Return the share of total units sold by the top ``fraction`` of rows.

    Parameters
    ----------
    sales : pandas.Series
        Observed, non-negative sales.
    fraction : float
        Fraction of rows in :math:`(0, 1]`, ranked by sales. At least one row
        is always included.

    Returns
    -------
    float
        :math:`\\sum_{i \\le k} s_{(i)} / \\sum_i s_i` for the :math:`k` largest
        values :math:`s_{(1)} \\ge s_{(2)} \\ge \\dots`.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    values = _observed(sales)
    total = values.sum()
    if total == 0:
        raise ValueError("sales sum to zero")
    k = max(1, int(np.floor(fraction * values.size)))
    return float(np.sort(values)[::-1][:k].sum() / total)


def _observed(sales: pd.Series) -> np.ndarray:
    values = np.asarray(sales, dtype="float64")
    if values.size == 0:
        raise ValueError("sales is empty")
    if np.isnan(values).any():
        raise ValueError("sales must not contain missing values")
    if (values < 0).any():
        raise ValueError("sales must be non-negative")
    return values


def _describe(values: np.ndarray) -> dict[str, float]:
    std = values.std(ddof=1) if values.size > 1 else np.nan
    centred = values - values.mean()
    scale = values.std(ddof=0)
    skew = float(np.mean(centred**3) / scale**3) if scale > 0 else np.nan
    stats = {
        "count": float(values.size),
        "mean": float(values.mean()),
        "std": float(std),
        "min": float(values.min()),
    }
    for q in QUANTILES:
        stats[f"q{round(q * 100):02d}"] = float(np.quantile(values, q))
    stats["max"] = float(values.max())
    stats["skewness"] = skew
    return stats


def target_summary(sales: pd.Series) -> pd.DataFrame:
    """Summarise observed sales on the raw and the ``log1p`` scale.

    Parameters
    ----------
    sales : pandas.Series
        Observed, non-negative sales in millions of units.

    Returns
    -------
    pandas.DataFrame
        Rows ``count``, ``mean``, ``std``, ``min``, quantiles ``q25`` to
        ``q99``, ``max`` and ``skewness`` (the moment coefficient
        :math:`m_3 / m_2^{3/2}`); columns ``raw`` and ``log1p``.
    """
    values = _observed(sales)
    return pd.DataFrame(
        {"raw": _describe(values), "log1p": _describe(np.log1p(values))}
    )


def censoring_summary(frame: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Compare observed sales of censored and uncensored releases.

    Parameters
    ----------
    frame : pandas.DataFrame
        Cleaned, release-scoped frame with ``censored`` and the target.
    target : str, default "total_sales"
        Target column.

    Returns
    -------
    pandas.DataFrame
        Index ``censored`` (False, True); columns ``n_rows``,
        ``median_sales`` and ``mean_log_sales`` (mean of ``log1p``).
    """
    _require(frame, [CENSORED, target])
    sales = frame[target]
    flag = frame[CENSORED].astype(bool)
    grouped = sales.groupby(flag)
    out = pd.DataFrame(
        {
            "n_rows": grouped.count().astype("int64"),
            "median_sales": grouped.median(),
            "mean_log_sales": np.log1p(sales).groupby(flag).mean(),
        }
    ).reindex([False, True])
    out["n_rows"] = out["n_rows"].fillna(0).astype("int64")
    out.index.name = CENSORED
    return out
