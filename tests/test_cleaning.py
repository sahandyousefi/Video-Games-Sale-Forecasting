import numpy as np
import pandas as pd
import pytest

from vgsales.data import schema
from vgsales.data.cleaning import clean, normalise_text
from vgsales.data.synthetic import SyntheticSpec, make_synthetic


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return make_synthetic(SyntheticSpec(n_titles=150), seed=11)


def test_untracked_rows_are_dropped_not_zero_filled(raw: pd.DataFrame) -> None:
    frame, report = clean(raw)
    assert frame["total_sales"].notna().all()
    assert len(frame) == raw["total_sales"].notna().sum()
    assert report.n_untracked == raw["total_sales"].isna().sum()


def test_target_values_are_unchanged(raw: pd.DataFrame) -> None:
    frame, _ = clean(raw)
    expected = raw["total_sales"].dropna().to_numpy()
    np.testing.assert_array_equal(frame["total_sales"].to_numpy(), expected)


def test_dates_are_parsed(raw: pd.DataFrame) -> None:
    frame, _ = clean(raw)
    assert schema.check_dtypes(frame) == {}
    assert pd.api.types.is_datetime64_any_dtype(frame["release_date"])


def test_normalise_text_handles_whitespace_and_empty_values() -> None:
    nbsp = chr(0xA0)
    series = pd.Series(
        ["  Nintendo   EAD ", f"Sony{nbsp}Computer", "   ", np.nan, "Sega"]
    )
    out = normalise_text(series)
    assert out.tolist()[:2] == ["Nintendo EAD", "Sony Computer"]
    assert out.isna().tolist() == [False, False, True, True, False]
    assert out[4] == "Sega"


def test_normalise_text_preserves_case() -> None:
    assert normalise_text(pd.Series(["id Software"]))[0] == "id Software"


def test_exact_duplicates_are_dropped(raw: pd.DataFrame) -> None:
    tracked = raw.dropna(subset=["total_sales"])
    doubled = pd.concat([raw, tracked.head(5)], ignore_index=True)
    frame, report = clean(doubled)
    assert report.n_duplicates == 5
    assert len(frame) == len(clean(raw)[0])


def test_whitespace_variants_count_as_duplicates(raw: pd.DataFrame) -> None:
    row = raw.dropna(subset=["total_sales"]).head(1).copy()
    row["publisher"] = "  " + row["publisher"] + " "
    _, report = clean(pd.concat([raw, row], ignore_index=True))
    assert report.n_duplicates == 1


def test_conflicting_keys_are_reported_and_kept(raw: pd.DataFrame) -> None:
    row = raw.dropna(subset=["total_sales"]).head(1).copy()
    row["total_sales"] = row["total_sales"] + 1.0
    frame, report = clean(pd.concat([raw, row], ignore_index=True))
    assert report.n_conflicting_keys == 2
    assert report.n_duplicates == 0
    assert len(frame) == len(clean(raw)[0]) + 1


def test_report_counts_are_consistent(raw: pd.DataFrame) -> None:
    frame, report = clean(raw)
    assert report.n_input == len(raw)
    assert report.n_output == len(frame)
    assert report.n_input - report.n_untracked - report.n_duplicates == len(frame)


def test_input_frame_is_not_modified(raw: pd.DataFrame) -> None:
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_output_has_fresh_index(raw: pd.DataFrame) -> None:
    frame, _ = clean(raw)
    pd.testing.assert_index_equal(frame.index, pd.RangeIndex(len(frame)))
