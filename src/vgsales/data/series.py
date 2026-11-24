r"""Monthly aggregate sales series by genre and platform family.

The title-level task predicts the lifetime sales of one release. The
aggregate view asks how many units the releases of a market segment in a
given month will sell, which is the quantity a publisher plans a release
calendar against. For a segment :math:`k` (a genre and platform family
pair) and a calendar month :math:`t`, let :math:`R_{k,t}` be the set of
tracked titles of the segment released in month :math:`t`. The series are

.. math::

    n_{k,t} = |R_{k,t}|, \qquad
    S_{k,t} = \sum_{i \in R_{k,t}} y_i,

where :math:`y_i` is the lifetime unit sales (``total_sales``) of title
:math:`i`. Only rows with an observed target enter; at the title level a
missing value means "not tracked" and is never read as zero. At the
aggregate level a month without tracked releases has :math:`n_{k,t} = 0`
and :math:`S_{k,t} = 0` by definition, and the monthly grid of every segment
is completed with such zeros between the common start and end months.

:math:`S_{k,t}` inherits the right-censoring of its titles: releases close
to the snapshot date have not accumulated their lifetime sales.
:func:`uncensored_end` returns the last month whose releases all lie
outside the censoring window of :func:`vgsales.data.temporal.flag_censored`,
and series meant for model fitting should end there.

Consoles are grouped into manufacturer families by :data:`PLATFORM_FAMILIES`,
so that a series spans console generations (PS2 to PS5, for example) rather
than ending with each platform's lifecycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from vgsales.data.schema import TARGET

PLATFORM_FAMILY: Final = "platform_family"
PERIOD: Final = "period"
N_TITLES: Final = "n_titles"
OTHER_FAMILY: Final = "other"
SERIES_KEYS: Final = ("genre", PLATFORM_FAMILY)

PLATFORM_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "playstation": ("PS", "PS2", "PS3", "PS4", "PS5", "PSP", "PSV", "PSN"),
    "xbox": ("XB", "X360", "XOne", "XS", "XBL"),
    "nintendo": (
        "NES",
        "SNES",
        "N64",
        "GC",
        "Wii",
        "WiiU",
        "NS",
        "GB",
        "GBC",
        "GBA",
        "DS",
        "DSi",
        "DSiW",
        "3DS",
        "VB",
    ),
    "sega": ("SG", "MS", "GEN", "SCD", "32X", "SAT", "DC", "GG"),
    "pc": ("PC", "OSX", "Linux"),
}

_FAMILY_OF: Final = {
    console.lower(): family
    for family, consoles in PLATFORM_FAMILIES.items()
    for console in consoles
}


def platform_family(console: pd.Series) -> pd.Series:
    """Map console codes to manufacturer families.

    Parameters
    ----------
    console : pandas.Series
        VGChartz console codes such as ``"PS4"`` or ``"X360"``. Matching
        ignores case and surrounding whitespace.

    Returns
    -------
    pandas.Series
        Family names from :data:`PLATFORM_FAMILIES`, and
        :data:`OTHER_FAMILY` for unlisted or missing consoles.
    """
    keys = console.astype(object).where(console.notna(), "").astype(str)
    families = keys.str.strip().str.lower().map(_FAMILY_OF)
    return families.fillna(OTHER_FAMILY).astype(object).rename(PLATFORM_FAMILY)


def release_month(release_date: pd.Series) -> pd.Series:
    """Return the first day of the release month of each date."""
    return release_date.dt.to_period("M").dt.to_timestamp().rename(PERIOD)


def uncensored_end(snapshot: pd.Timestamp | str, window_years: int) -> pd.Timestamp:
    """Return the last month whose releases are all outside the censoring window.

    Parameters
    ----------
    snapshot : pandas.Timestamp or str
        Snapshot date of the extract.
    window_years : int
        ``data.censoring_window_years`` of the configuration.

    Returns
    -------
    pandas.Timestamp
        First day of the month before the month of the censoring cutoff
        ``snapshot - window_years``, so every release of that month and of
        earlier months precedes the cutoff.
    """
    if window_years < 0:
        raise ValueError("window_years must be >= 0")
    cutoff = pd.Timestamp(snapshot) - pd.DateOffset(years=window_years)
    return (cutoff.to_period("M") - 1).to_timestamp()


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.period_range(start, end, freq="M").to_timestamp()


def aggregate_series(
    frame: pd.DataFrame,
    by: Sequence[str] = SERIES_KEYS,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    min_titles: int = 1,
) -> pd.DataFrame:
    """Aggregate title-level sales into monthly series per segment.

    Parameters
    ----------
    frame : pandas.DataFrame
        Cleaned title-level frame with ``release_date``, ``total_sales`` and
        the columns of ``by``. ``platform_family`` is derived from
        ``console`` when it is not present.
    by : sequence of str, default ("genre", "platform_family")
        Columns that define a segment. Missing values form their own
        segment, labelled ``"__missing__"``.
    start, end : timestamp or str, optional
        First and last month of the grid. Default to the first and last
        release month in ``frame``. Releases outside the range are dropped.
    min_titles : int, default 1
        Minimum number of titles of a segment over the whole grid; sparser
        segments are dropped.

    Returns
    -------
    pandas.DataFrame
        Long frame with the columns of ``by``, ``period`` (first day of the
        month), ``n_titles`` and ``total_sales``, with one row per segment
        and month, sorted by segment and month.

    Raises
    ------
    ValueError
        If a column is missing, a target value is missing or negative, or
        the range is empty.
    """
    by = list(by)
    if not by:
        raise ValueError("by must name at least one column")
    if min_titles < 1:
        raise ValueError("min_titles must be >= 1")
    data = frame
    if PLATFORM_FAMILY in by and PLATFORM_FAMILY not in data.columns:
        if "console" not in data.columns:
            raise ValueError("frame needs 'console' to derive the platform family")
        data = data.assign(**{PLATFORM_FAMILY: platform_family(data["console"])})
    missing = sorted({"release_date", TARGET, *by} - set(data.columns))
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    sales = data[TARGET].to_numpy(dtype=np.float64)
    if np.isnan(sales).any() or (sales < 0).any():
        raise ValueError(
            f"{TARGET} must be observed and non-negative; drop untracked titles "
            "before aggregating"
        )
    dated = data.loc[data["release_date"].notna()]
    month = release_month(dated["release_date"])
    first = pd.Timestamp(start) if start is not None else month.min()
    last = pd.Timestamp(end) if end is not None else month.max()
    if pd.isna(first) or pd.isna(last):
        raise ValueError("frame has no dated releases")
    first = first.to_period("M").to_timestamp()
    last = last.to_period("M").to_timestamp()
    if first > last:
        raise ValueError(f"start {first.date()} is after end {last.date()}")
    inside = ((month >= first) & (month <= last)).to_numpy()

    keys = dated.loc[inside, by].astype(object)
    keys = keys.where(keys.notna(), "__missing__")
    grouped = (
        keys.assign(**{PERIOD: month[inside], TARGET: dated.loc[inside, TARGET]})
        .groupby([*by, PERIOD], sort=True)[TARGET]
        .agg(["size", "sum"])
        .rename(columns={"size": N_TITLES, "sum": TARGET})
    )
    segments = grouped.groupby(level=by)[N_TITLES].sum()
    segments = segments.index[segments >= min_titles]
    if len(segments) == 0:
        raise ValueError(f"no segment has {min_titles} or more titles in the range")
    # A single grouping column yields scalar labels on some pandas versions.
    segment_tuples = [s if isinstance(s, tuple) else (s,) for s in segments]
    grid = pd.MultiIndex.from_tuples(
        [(*s, p) for s in segment_tuples for p in _month_range(first, last)],
        names=[*by, PERIOD],
    )
    out = grouped.reindex(grid, fill_value=0).reset_index()
    out[N_TITLES] = out[N_TITLES].astype(np.int64)
    out[TARGET] = out[TARGET].astype(np.float64)
    return out


def to_wide(series: pd.DataFrame, value: str = TARGET) -> pd.DataFrame:
    """Pivot the long output of :func:`aggregate_series` to one column per segment.

    Parameters
    ----------
    series : pandas.DataFrame
        Output of :func:`aggregate_series`.
    value : {"total_sales", "n_titles"}, default "total_sales"
        Column to pivot.

    Returns
    -------
    pandas.DataFrame
        Months as rows (``period`` index) and segments as columns, labelled
        by the segment values joined with ``"/"``.
    """
    keys = [c for c in series.columns if c not in (PERIOD, N_TITLES, TARGET)]
    if value not in (N_TITLES, TARGET) or not keys:
        raise ValueError("series must be the output of aggregate_series")
    label = series[keys].astype(str).agg("/".join, axis=1)
    wide = series.assign(segment=label).pivot(
        index=PERIOD, columns="segment", values=value
    )
    wide.columns.name = None
    return wide
