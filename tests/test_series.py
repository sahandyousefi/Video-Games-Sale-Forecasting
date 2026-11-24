import numpy as np
import pandas as pd
import pytest

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.schema import TARGET
from vgsales.data.series import (
    N_TITLES,
    OTHER_FAMILY,
    PERIOD,
    PLATFORM_FAMILY,
    aggregate_series,
    platform_family,
    to_wide,
    uncensored_end,
)
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope, flag_censored


@pytest.fixture(scope="module")
def titles() -> pd.DataFrame:
    raw = make_synthetic(SyntheticSpec(n_titles=1500), seed=8)
    return apply_release_scope(clean(raw)[0], load_config().data)[0]


def test_platform_families() -> None:
    consoles = pd.Series(["PS4", "x360", " NS ", "PC", "DC", "Mob", None])
    assert platform_family(consoles).tolist() == [
        "playstation",
        "xbox",
        "nintendo",
        "pc",
        "sega",
        OTHER_FAMILY,
        OTHER_FAMILY,
    ]


def test_aggregates_preserve_titles_and_sales(titles: pd.DataFrame) -> None:
    series = aggregate_series(titles)
    assert list(series.columns) == ["genre", PLATFORM_FAMILY, PERIOD, N_TITLES, TARGET]
    assert series[N_TITLES].sum() == len(titles)
    assert series[TARGET].sum() == pytest.approx(titles[TARGET].sum())

    families = platform_family(titles["console"])
    mask = (titles["genre"] == "Action") & (families == "playstation")
    month = titles["release_date"].dt.to_period("M").dt.to_timestamp()
    expected = titles.loc[mask, TARGET].groupby(month[mask]).sum()
    cell = series[
        (series["genre"] == "Action") & (series[PLATFORM_FAMILY] == "playstation")
    ]
    observed = cell.set_index(PERIOD)[TARGET]
    pd.testing.assert_series_equal(
        observed.loc[expected.index], expected, check_names=False
    )
    assert observed.drop(expected.index).eq(0).all()


def test_every_segment_has_a_complete_monthly_grid(titles: pd.DataFrame) -> None:
    series = aggregate_series(titles)
    first = titles["release_date"].min().to_period("M")
    last = titles["release_date"].max().to_period("M")
    n_months = (last - first).n + 1
    sizes = series.groupby(["genre", PLATFORM_FAMILY]).size()
    assert (sizes == n_months).all()
    periods = series.groupby(["genre", PLATFORM_FAMILY])[PERIOD]
    assert periods.apply(lambda p: p.is_monotonic_increasing and p.is_unique).all()
    assert ((series[N_TITLES] == 0) == (series[TARGET] == 0)).all()


def test_range_and_sparse_segments(titles: pd.DataFrame) -> None:
    series = aggregate_series(titles, start="2005-03-17", end="2010-12-31")
    assert series[PERIOD].min() == pd.Timestamp("2005-03-01")
    assert series[PERIOD].max() == pd.Timestamp("2010-12-01")
    inside = titles["release_date"].between("2005-03-01", "2010-12-31")
    assert series[N_TITLES].sum() == inside.sum()

    dense = aggregate_series(titles, min_titles=15)
    totals = dense.groupby(["genre", PLATFORM_FAMILY])[N_TITLES].sum()
    assert (totals >= 15).all()
    assert len(totals) < len(
        aggregate_series(titles).groupby(["genre", PLATFORM_FAMILY])
    )


def test_single_key_and_wide_layout(titles: pd.DataFrame) -> None:
    series = aggregate_series(titles, by=("genre",))
    wide = to_wide(series)
    assert wide.shape == (series[PERIOD].nunique(), titles["genre"].nunique())
    assert wide.sum().sum() == pytest.approx(titles[TARGET].sum())
    counts = to_wide(aggregate_series(titles), value=N_TITLES)
    assert "Action/playstation" in counts.columns
    assert counts.to_numpy().sum() == len(titles)


def test_uncensored_end_excludes_the_censoring_window(titles: pd.DataFrame) -> None:
    assert uncensored_end("2024-06-15", 2) == pd.Timestamp("2022-05-01")
    assert uncensored_end("2024-06-30", 0) == pd.Timestamp("2024-05-01")
    snapshot = titles["release_date"].max()
    end = uncensored_end(snapshot, 2)
    series = aggregate_series(titles, end=end)
    month = titles["release_date"].dt.to_period("M").dt.to_timestamp()
    kept = titles[month <= end]
    assert not flag_censored(kept["release_date"], snapshot, 2).any()
    assert series[N_TITLES].sum() == len(kept)


def test_invalid_inputs_are_rejected(titles: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="observed"):
        aggregate_series(titles.assign(**{TARGET: np.nan}))
    with pytest.raises(ValueError, match="missing columns"):
        aggregate_series(titles.drop(columns="genre"))
    with pytest.raises(ValueError, match="console"):
        aggregate_series(titles.drop(columns="console"))
    with pytest.raises(ValueError, match="after end"):
        aggregate_series(titles, start="2010-01-01", end="2009-01-01")
    with pytest.raises(ValueError, match="no segment"):
        aggregate_series(titles, min_titles=10**6)
