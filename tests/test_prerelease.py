import numpy as np
import pandas as pd
import pytest

from vgsales.config import load_config
from vgsales.data.cleaning import clean
from vgsales.data.schema import LEAKAGE_COLUMNS, REGIONAL_SALES, TARGET
from vgsales.data.synthetic import SyntheticSpec, make_synthetic
from vgsales.data.temporal import apply_release_scope
from vgsales.features.prerelease import (
    NUMERIC_FEATURES,
    OUTPUT_COLUMNS,
    build_prerelease_features,
    calendar_features,
    platform_age,
    platform_counts,
)


def _rows(records: list[tuple[str, str, str]]) -> pd.DataFrame:
    titles, consoles, dates = zip(*records, strict=True)
    n = len(records)
    return pd.DataFrame(
        {
            "title": list(titles),
            "console": list(consoles),
            "genre": ["Action"] * n,
            "publisher": ["P"] * n,
            "developer": ["D"] * n,
            "critic_score": [8.0] + [np.nan] * (n - 1),
            "total_sales": np.linspace(0.1, 1.0, n),
            "na_sales": np.full(n, 0.05),
            "release_date": pd.to_datetime(list(dates)),
        }
    )


@pytest.fixture
def small() -> pd.DataFrame:
    return _rows(
        [
            ("Game A", "PS4", "2015-03-10"),
            ("Game: A", "XOne", "2015-11-20"),
            ("GAME A", "PC", "2017-06-01"),
            ("Other", "PS4", "2013-11-15"),
            ("Solo", "NS", "2017-03-03"),
        ]
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    raw = make_synthetic(SyntheticSpec(n_titles=500), seed=5)
    scoped, _ = apply_release_scope(clean(raw)[0], load_config().data)
    return scoped


def test_calendar_features() -> None:
    dates = pd.Series(pd.to_datetime(["2010-11-15", "2010-03-01", "2011-10-31"]))
    out = calendar_features(dates)
    assert out["release_month"].tolist() == [11, 3, 10]
    assert out["release_quarter"].tolist() == [4, 1, 4]
    assert out["holiday_season"].tolist() == [1, 0, 0]
    with pytest.raises(ValueError, match="missing"):
        calendar_features(pd.Series(pd.to_datetime(["2010-01-01", None])))


def test_platform_age_counts_from_first_release_on_console(
    small: pd.DataFrame,
) -> None:
    age = platform_age(small, small)
    # PS4 first appears on 2013-11-15; the earliest row of each console is 0.
    days = (pd.Timestamp("2015-03-10") - pd.Timestamp("2013-11-15")).days
    assert age.iloc[0] == pytest.approx(days / 365.25)
    assert age.iloc[[1, 2, 3, 4]].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_platform_age_without_earlier_reference_rows_is_zero(
    small: pd.DataFrame,
) -> None:
    reference = small.iloc[[2]]
    age = platform_age(small, reference)
    assert (age == 0.0).all()
    missing = small.assign(console=[None, "XOne", "PC", "PS4", "NS"])
    assert np.isnan(platform_age(missing, small).iloc[0])


def test_platform_counts_use_same_or_earlier_years(small: pd.DataFrame) -> None:
    out = platform_counts(small, small)
    assert out["n_platforms"].tolist() == [2, 2, 3, 1, 1]
    assert out["is_port"].tolist() == [0, 0, 1, 0, 0]


def test_platform_counts_with_wider_reference(small: pd.DataFrame) -> None:
    untracked = _rows([("Solo", "PC", "2016-12-01")])
    reference = pd.concat([small, untracked], ignore_index=True)
    out = platform_counts(small, reference)
    assert out.loc[4, "n_platforms"] == 2
    assert out.loc[4, "is_port"] == 1


def test_builder_output_layout(small: pd.DataFrame) -> None:
    out = build_prerelease_features(small.set_index(np.arange(10, 15)))
    assert tuple(out.columns) == OUTPUT_COLUMNS
    assert out.index.tolist() == list(range(10, 15))
    assert out["release_year"].tolist() == [2015, 2015, 2017, 2013, 2017]
    assert out["critic_missing"].tolist() == [0, 1, 1, 1, 1]
    assert out["title_key"].nunique() == 3
    for col in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(out[col])


def test_builder_excludes_leakage_and_target(frame: pd.DataFrame) -> None:
    out = build_prerelease_features(frame)
    assert not set(out.columns) & {TARGET, *LEAKAGE_COLUMNS}
    perturbed = frame.copy()
    rng = np.random.default_rng(0)
    for col in (TARGET, *REGIONAL_SALES):
        perturbed[col] = rng.permutation(perturbed[col].to_numpy())
    perturbed["last_update"] = pd.NaT
    pd.testing.assert_frame_equal(build_prerelease_features(perturbed), out)


def test_future_rows_leave_past_features_unchanged(frame: pd.DataFrame) -> None:
    past = frame[frame["release_year"] <= 2012]
    on_past = build_prerelease_features(past)
    on_all = build_prerelease_features(frame).loc[past.index]
    pd.testing.assert_frame_equal(on_all, on_past)


def test_builder_validates_input(small: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="critic_score"):
        build_prerelease_features(small.drop(columns="critic_score"))
    with pytest.raises(TypeError, match="datetime"):
        build_prerelease_features(small.assign(release_date="2015-01-01"))
    with pytest.raises(ValueError, match="missing values"):
        build_prerelease_features(small.assign(release_date=pd.NaT))
    with pytest.raises(ValueError, match="reference"):
        build_prerelease_features(small, reference=small[["title", "console"]])
