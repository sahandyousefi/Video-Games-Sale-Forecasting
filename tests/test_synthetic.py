import numpy as np
import pandas as pd
import pytest

from vgsales.data import schema
from vgsales.data.synthetic import PLATFORM_WINDOWS, SyntheticSpec, make_synthetic


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_synthetic(SyntheticSpec(n_titles=300), seed=1)


def test_columns_follow_raw_schema(frame: pd.DataFrame) -> None:
    assert tuple(frame.columns) == schema.RAW_COLUMNS
    assert schema.check_columns(frame).ok


def test_same_seed_gives_identical_frames() -> None:
    spec = SyntheticSpec(n_titles=50)
    pd.testing.assert_frame_equal(make_synthetic(spec, 3), make_synthetic(spec, 3))


def test_different_seeds_differ() -> None:
    spec = SyntheticSpec(n_titles=50)
    assert not make_synthetic(spec, 3).equals(make_synthetic(spec, 4))


def test_values_respect_schema_ranges(frame: pd.DataFrame) -> None:
    assert schema.check_ranges(frame) == {}


def test_missingness_resembles_real_extract(frame: pd.DataFrame) -> None:
    missing = frame.isna().mean()
    assert 0.45 < missing["total_sales"] < 0.85
    assert missing["critic_score"] > 0.7
    assert 0.03 < missing["release_date"] < 0.2
    assert missing["na_sales"] >= missing["total_sales"]


def test_regional_sales_sum_to_total_up_to_rounding(frame: pd.DataFrame) -> None:
    regional = frame[list(schema.REGIONAL_SALES)]
    complete = regional.notna().all(axis=1)
    gap = (regional[complete].sum(axis=1) - frame.loc[complete, "total_sales"]).abs()
    assert complete.any()
    assert (gap <= 0.025).all()


def test_titles_span_several_platforms(frame: pd.DataFrame) -> None:
    per_title = frame.groupby("title")["console"]
    assert (per_title.size() > 1).any()
    assert (per_title.nunique() == per_title.size()).all()


def test_release_years_fall_in_platform_windows(frame: pd.DataFrame) -> None:
    dated = frame.dropna(subset=["release_date"])
    years = pd.to_datetime(dated["release_date"]).dt.year
    first = dated["console"].map(lambda c: PLATFORM_WINDOWS[c][0])
    last = dated["console"].map(lambda c: PLATFORM_WINDOWS[c][1])
    assert ((years >= first) & (years <= last)).all()


def test_last_update_never_precedes_release(frame: pd.DataFrame) -> None:
    both = frame.dropna(subset=["release_date", "last_update"])
    release = pd.to_datetime(both["release_date"])
    update = pd.to_datetime(both["last_update"])
    assert (update >= release).all()


def test_sales_missing_rows_have_no_regional_values(frame: pd.DataFrame) -> None:
    untracked = frame["total_sales"].isna()
    assert frame.loc[untracked, list(schema.REGIONAL_SALES)].isna().all().all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_titles": 0},
        {"start_year": 2010, "end_year": 2005},
        {"start_year": 1970},
        {"p_sales_observed": 1.5},
        {"snapshot": "2024-02-30"},
    ],
)
def test_invalid_spec_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SyntheticSpec(**kwargs)


def test_critic_scores_are_one_decimal(frame: pd.DataFrame) -> None:
    scores = frame["critic_score"].dropna().to_numpy()
    assert np.allclose(scores, np.round(scores, 1))
