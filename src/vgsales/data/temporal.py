"""Release-year scope and right-censoring relative to the data snapshot.

Sales in the extract are cumulative up to the snapshot date :math:`s`. For a
title released at :math:`r`, the observed value is the sales accumulated over
:math:`[r, s]`, which understates lifetime sales when :math:`s - r` is short.
Rows with :math:`r > s - w`, where :math:`w` is
``data.censoring_window_years``, are flagged as right-censored so that
evaluation can report them separately.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

from vgsales.config import DataConfig

logger = logging.getLogger(__name__)

RELEASE_YEAR = "release_year"
CENSORED = "censored"


@dataclass(frozen=True)
class TemporalReport:
    """Snapshot date and row counts removed by the release-year filter.

    Attributes
    ----------
    snapshot : str
        Snapshot date in ISO format.
    n_input : int
        Rows received.
    n_missing_release : int
        Rows dropped because ``release_date`` is missing.
    n_before_min : int
        Rows dropped with a release year before ``min_release_year``.
    n_after_max : int
        Rows dropped with a release year after ``max_release_year``.
    n_after_snapshot : int
        Rows dropped with a release date after the snapshot.
    n_output : int
        Rows returned.
    n_censored : int
        Returned rows flagged as right-censored.
    """

    snapshot: str
    n_input: int
    n_missing_release: int
    n_before_min: int
    n_after_max: int
    n_after_snapshot: int
    n_output: int
    n_censored: int

    def to_dict(self) -> dict[str, int | str]:
        """Return the report as a plain dictionary."""
        return asdict(self)


def resolve_snapshot(df: pd.DataFrame, snapshot_date: str | None) -> pd.Timestamp:
    """Return the snapshot date of the extract.

    The configured date takes precedence. Otherwise the latest ``last_update``
    is used. ``last_update`` is never a model input; it is read here only as
    dataset-level metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with a datetime ``last_update`` column.
    snapshot_date : str or None
        ISO date from ``data.snapshot_date``.

    Returns
    -------
    pandas.Timestamp
        Snapshot date.

    Raises
    ------
    ValueError
        If no date is configured and ``last_update`` has no observed values.
    """
    if snapshot_date is not None:
        return pd.Timestamp(snapshot_date)
    latest = df["last_update"].max()
    if pd.isna(latest):
        raise ValueError(
            "snapshot date cannot be inferred: last_update has no observed values; "
            "set data.snapshot_date"
        )
    return pd.Timestamp(latest)


def flag_censored(
    release_date: pd.Series, snapshot: pd.Timestamp, window_years: int
) -> pd.Series:
    """Flag releases within ``window_years`` of the snapshot.

    Parameters
    ----------
    release_date : pandas.Series
        Datetime release dates.
    snapshot : pandas.Timestamp
        Snapshot date.
    window_years : int
        Length of the censoring window in calendar years. Zero flags no row.

    Returns
    -------
    pandas.Series
        Boolean series, True where ``release_date > snapshot - window_years``.
    """
    if window_years == 0:
        return pd.Series(False, index=release_date.index, name=CENSORED)
    cutoff = snapshot - pd.DateOffset(years=window_years)
    return (release_date > cutoff).rename(CENSORED)


def apply_release_scope(
    df: pd.DataFrame, config: DataConfig
) -> tuple[pd.DataFrame, TemporalReport]:
    """Restrict rows to the configured release years and flag censoring.

    Rows are dropped when the release date is missing (they cannot be placed
    on the time axis used for splitting), when the release year lies outside
    ``[min_release_year, max_release_year]``, or when the release date is
    after the snapshot. Two columns are added: ``release_year`` (int64) and
    ``censored`` (bool).

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned frame with datetime ``release_date`` and ``last_update``.
    config : DataConfig
        Data section of the pipeline configuration.

    Returns
    -------
    frame : pandas.DataFrame
        Filtered frame with a fresh ``RangeIndex``.
    report : TemporalReport
        Snapshot date and counts for each filter.
    """
    snapshot = resolve_snapshot(df, config.snapshot_date)
    release = df["release_date"]
    year = release.dt.year

    missing = release.isna()
    before_min = ~missing & (year < config.min_release_year)
    after_max = pd.Series(False, index=df.index)
    if config.max_release_year is not None:
        after_max = ~missing & (year > config.max_release_year)
    after_snapshot = ~missing & ~before_min & ~after_max & (release > snapshot)

    keep = ~(missing | before_min | after_max | after_snapshot)
    frame = df.loc[keep].reset_index(drop=True)
    frame[RELEASE_YEAR] = frame["release_date"].dt.year.astype("int64")
    frame[CENSORED] = flag_censored(
        frame["release_date"], snapshot, config.censoring_window_years
    ).astype(bool)

    report = TemporalReport(
        snapshot=snapshot.date().isoformat(),
        n_input=len(df),
        n_missing_release=int(missing.sum()),
        n_before_min=int(before_min.sum()),
        n_after_max=int(after_max.sum()),
        n_after_snapshot=int(after_snapshot.sum()),
        n_output=len(frame),
        n_censored=int(frame[CENSORED].sum()),
    )
    logger.info("release scope: %s", report.to_dict())
    return frame, report
