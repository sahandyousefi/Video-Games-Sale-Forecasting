import pandas as pd
import pytest

from vgsales.config import DataConfig
from vgsales.data.cleaning import clean
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import (
    CENSORED,
    RELEASE_YEAR,
    apply_release_scope,
    flag_censored,
    resolve_snapshot,
)


@pytest.fixture(scope="module")
def cleaned() -> pd.DataFrame:
    raw = make_synthetic(SyntheticSpec(n_titles=200, end_year=2022), seed=5)
    return clean(raw)[0]


def _dated(dates: list[str | None], updates: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "release_date": pd.to_datetime(pd.Series(dates)),
            "last_update": pd.to_datetime(pd.Series(updates)),
            "total_sales": 1.0,
        }
    )


def test_configured_snapshot_takes_precedence() -> None:
    df = _dated(["2010-01-01"], ["2020-05-01"])
    assert resolve_snapshot(df, "2021-01-31") == pd.Timestamp("2021-01-31")


def test_snapshot_defaults_to_latest_update() -> None:
    df = _dated(["2010-01-01", "2011-01-01"], ["2020-05-01", "2022-03-15"])
    assert resolve_snapshot(df, None) == pd.Timestamp("2022-03-15")


def test_snapshot_without_updates_raises() -> None:
    df = _dated(["2010-01-01"], [None])
    with pytest.raises(ValueError, match="snapshot_date"):
        resolve_snapshot(df, None)


def test_censoring_boundary_is_exclusive() -> None:
    dates = pd.to_datetime(pd.Series(["2022-06-30", "2022-07-01", "2015-01-01"]))
    flags = flag_censored(dates, pd.Timestamp("2024-06-30"), window_years=2)
    assert flags.tolist() == [False, True, False]
    assert flags.name == CENSORED


def test_zero_window_flags_nothing() -> None:
    dates = pd.to_datetime(pd.Series(["2024-06-29"]))
    assert not flag_censored(dates, pd.Timestamp("2024-06-30"), 0).any()


def test_rows_outside_scope_are_dropped_and_counted() -> None:
    df = _dated(
        ["1975-03-01", "1995-01-01", None, "2019-05-05", "2023-02-01", "2025-01-01"],
        [None, "2020-01-01", None, "2023-12-31", "2024-01-01", None],
    )
    config = DataConfig(
        min_release_year=1980, max_release_year=2023, snapshot_date="2024-01-01"
    )
    frame, report = apply_release_scope(df, config)
    assert frame[RELEASE_YEAR].tolist() == [1995, 2019, 2023]
    assert report.n_missing_release == 1
    assert report.n_before_min == 1
    assert report.n_after_max == 1
    assert report.n_after_snapshot == 0
    assert report.n_output == 3
    assert frame[CENSORED].tolist() == [False, False, True]


def test_releases_after_snapshot_are_dropped() -> None:
    df = _dated(["2020-01-01", "2024-03-01"], ["2024-01-01", None])
    frame, report = apply_release_scope(df, DataConfig())
    assert report.snapshot == "2024-01-01"
    assert report.n_after_snapshot == 1
    assert len(frame) == 1


def test_synthetic_scope(cleaned: pd.DataFrame) -> None:
    config = DataConfig(snapshot_date="2024-06-30", censoring_window_years=2)
    frame, report = apply_release_scope(cleaned, config)
    assert frame["release_date"].notna().all()
    assert frame[RELEASE_YEAR].dtype == "int64"
    assert (frame[RELEASE_YEAR] == frame["release_date"].dt.year).all()
    censored = frame.loc[frame[CENSORED], "release_date"]
    assert (censored > pd.Timestamp("2022-06-30")).all()
    assert report.n_censored == len(censored)
    dropped = (
        report.n_missing_release
        + report.n_before_min
        + report.n_after_max
        + report.n_after_snapshot
    )
    assert report.n_input - dropped == report.n_output == len(frame)


def test_input_frame_is_not_modified(cleaned: pd.DataFrame) -> None:
    before = cleaned.copy()
    apply_release_scope(cleaned, DataConfig())
    pd.testing.assert_frame_equal(cleaned, before)
