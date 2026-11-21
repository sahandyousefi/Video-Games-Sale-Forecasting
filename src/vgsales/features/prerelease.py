"""Model inputs that are known at or before a title's release date.

Every feature of row :math:`i` is a function of the row's own attributes and
of the release metadata (title, console, release date) of rows released no
later than the calendar year of row :math:`i`. No target value and no column
in :data:`vgsales.data.schema.LEAKAGE_COLUMNS` is read: the builder selects
its inputs from an explicit allowlist, so regional sales cannot enter the
feature matrix (known issue 2 in ``ROADMAP.md``). Because no statistic is
learned from the target, the features can be computed once on the full frame
before splitting, and appending later releases leaves the features of earlier
rows unchanged.

Feature definitions, for a row with release date :math:`d`, release year
:math:`t`, console :math:`p` and title key :math:`g`:

- ``release_month``, ``release_quarter``: calendar position of :math:`d`.
- ``holiday_season``: 1 if :math:`d` falls in November or December.
- ``platform_age_years``: :math:`(d - L_p) / 365.25`, where :math:`L_p` is
  the earliest release date observed on console :math:`p`. It measures the
  position of the release in the platform lifecycle.
- ``n_platforms``: number of distinct consoles on which title :math:`g` was
  released in year :math:`t` or earlier, including :math:`p`.
- ``is_port``: 1 if title :math:`g` was released on another console in a
  year before :math:`t`.
- ``critic_score``: unchanged, with missing values kept for the fitted
  imputer of the preprocessing pipeline.
- ``critic_missing``: 1 if ``critic_score`` is missing. Coverage of critic
  scores is low and related to title size, so missingness is informative.

The platform features are estimated from the rows of a reference frame. By
default this is the frame itself, which after cleaning holds tracked titles
only; passing the loaded frame before cleaning as ``reference`` also counts
untracked releases.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from vgsales.data.grouping import title_key
from vgsales.data.schema import CATEGORICAL, assert_no_leakage
from vgsales.data.temporal import RELEASE_YEAR

GROUP_COLUMN: Final = "title_key"
HOLIDAY_MONTHS: Final = (11, 12)
DAYS_PER_YEAR: Final = 365.25

CALENDAR_FEATURES: Final = ("release_month", "release_quarter", "holiday_season")
PLATFORM_FEATURES: Final = ("platform_age_years", "n_platforms", "is_port")
CRITIC_FEATURES: Final = ("critic_score", "critic_missing")
NUMERIC_FEATURES: Final = (*CALENDAR_FEATURES, *PLATFORM_FEATURES, *CRITIC_FEATURES)
FEATURE_COLUMNS: Final = (*CATEGORICAL, RELEASE_YEAR, *NUMERIC_FEATURES)
OUTPUT_COLUMNS: Final = (*FEATURE_COLUMNS, GROUP_COLUMN)

_REFERENCE_COLUMNS: Final = ("title", "console", "release_date")
_INPUT_COLUMNS: Final = (*_REFERENCE_COLUMNS, *CATEGORICAL, "critic_score")


def _require(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(frame["release_date"]):
        raise TypeError(f"{name}.release_date must be a datetime column")


def calendar_features(release_date: pd.Series) -> pd.DataFrame:
    """Return month, quarter and holiday-season indicator of each release.

    Parameters
    ----------
    release_date : pandas.Series
        Datetime release dates without missing values.

    Returns
    -------
    pandas.DataFrame
        Integer columns ``release_month``, ``release_quarter`` and
        ``holiday_season`` with the index of ``release_date``.
    """
    if release_date.isna().any():
        raise ValueError("release_date must not contain missing values")
    month = release_date.dt.month
    return pd.DataFrame(
        {
            "release_month": month.astype("int64"),
            "release_quarter": release_date.dt.quarter.astype("int64"),
            "holiday_season": month.isin(HOLIDAY_MONTHS).astype("int64"),
        },
        index=release_date.index,
    )


def platform_age(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Return the years between each release and its platform's first release.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to describe, with ``console`` and datetime ``release_date``.
    reference : pandas.DataFrame
        Rows from which the earliest release date of each console is taken.

    Returns
    -------
    pandas.Series
        Float ``platform_age_years``. Zero when the reference has no earlier
        release on the console, ``NaN`` when the console is missing.
    """
    launch = reference.groupby("console")["release_date"].min()
    first = frame["console"].map(launch)
    date = frame["release_date"]
    # A reference without earlier releases on the console makes this row
    # the first one observed.
    first = first.where(first.notna() & (first <= date), date)
    age = (date - first).dt.days.astype("float64") / DAYS_PER_YEAR
    return age.where(frame["console"].notna()).rename("platform_age_years")


def platform_counts(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Count the platforms of each title up to the row's release year.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to describe, with ``title``, ``console`` and ``release_date``.
    reference : pandas.DataFrame
        Rows that define on which consoles, and from which year, each title
        was available.

    Returns
    -------
    pandas.DataFrame
        Integer columns ``n_platforms`` and ``is_port`` with the index of
        ``frame``. Rows without a usable title key count only their own
        platform.
    """
    first_year = (
        pd.DataFrame(
            {
                "key": title_key(reference["title"]).to_numpy(),
                "other": reference["console"].to_numpy(),
                "other_year": reference["release_date"].dt.year.to_numpy(),
            }
        )
        .dropna()
        .groupby(["key", "other"], as_index=False)["other_year"]
        .min()
    )
    rows = pd.DataFrame(
        {
            "row": np.arange(len(frame)),
            "key": title_key(frame["title"]).to_numpy(),
            "console": frame["console"].to_numpy(),
            "year": frame["release_date"].dt.year.to_numpy(),
        }
    )
    pairs = rows.merge(first_year, on="key", how="inner")
    pairs = pairs[pairs["other"] != pairs["console"]]
    earlier_or_same = pairs[pairs["other_year"] <= pairs["year"]]
    strictly_earlier = pairs[pairs["other_year"] < pairs["year"]]
    n_other = earlier_or_same.groupby("row").size()
    n_earlier = strictly_earlier.groupby("row").size()
    position = pd.RangeIndex(len(frame))
    n_platforms = 1 + n_other.reindex(position, fill_value=0).to_numpy()
    is_port = n_earlier.reindex(position, fill_value=0).to_numpy() > 0
    return pd.DataFrame(
        {
            "n_platforms": n_platforms.astype("int64"),
            "is_port": is_port.astype("int64"),
        },
        index=frame.index,
    )


def build_prerelease_features(
    frame: pd.DataFrame, reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Assemble the pre-release model inputs of each row.

    Parameters
    ----------
    frame : pandas.DataFrame
        Cleaned and release-scoped rows with the categorical columns,
        ``title``, ``critic_score`` and datetime ``release_date``.
    reference : pandas.DataFrame, optional
        Rows used to estimate platform launch dates and the platforms of each
        title. Must have ``title``, ``console`` and datetime
        ``release_date``; rows without a release date are ignored. Defaults
        to ``frame``.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`OUTPUT_COLUMNS` with the index of ``frame``: the
        categorical inputs, ``release_year``, the numeric features and the
        ``title_key`` group label used for splitting.

    Raises
    ------
    ValueError
        If required columns are missing or ``release_date`` has missing
        values.
    TypeError
        If ``release_date`` is not a datetime column.
    """
    _require(frame, _INPUT_COLUMNS, "frame")
    reference = frame if reference is None else reference
    _require(reference, _REFERENCE_COLUMNS, "reference")
    reference = reference.loc[reference["release_date"].notna()]

    calendar = calendar_features(frame["release_date"])
    critic = frame["critic_score"].astype("float64")
    out = pd.concat(
        [
            frame.loc[:, list(CATEGORICAL)],
            frame["release_date"].dt.year.astype("int64").rename(RELEASE_YEAR),
            calendar,
            platform_age(frame, reference),
            platform_counts(frame, reference),
            critic.rename("critic_score"),
            critic.isna().astype("int64").rename("critic_missing"),
            title_key(frame["title"]).rename(GROUP_COLUMN),
        ],
        axis=1,
    )
    assert_no_leakage(out.columns)
    return out.loc[:, list(OUTPUT_COLUMNS)]
