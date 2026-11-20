from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vgsales.data import schema
from vgsales.data.loading import SchemaError, coerce_raw, load_raw
from vgsales.data.synthetic import SyntheticSpec, make_synthetic


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return make_synthetic(SyntheticSpec(n_titles=120), seed=7)


@pytest.fixture
def csv_path(raw: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "vgchartz-2024.csv"
    raw.to_csv(path, index=False)
    return path


def test_load_raw_returns_schema_dtypes(csv_path: Path) -> None:
    frame = load_raw(csv_path)
    assert tuple(frame.columns) == schema.RAW_COLUMNS
    assert schema.check_dtypes(frame) == {}
    assert frame["release_date"].dtype == "datetime64[ns]"
    assert frame["total_sales"].dtype == "float64"
    assert frame["console"].dtype == object


def test_load_raw_preserves_values_and_missingness(
    raw: pd.DataFrame, csv_path: Path
) -> None:
    frame = load_raw(csv_path)
    assert len(frame) == len(raw)
    pd.testing.assert_series_equal(frame.isna().sum(), raw.isna().sum())
    np.testing.assert_allclose(frame["total_sales"], raw["total_sales"])
    expected = pd.to_datetime(raw["release_date"])
    assert (frame["release_date"].dropna() == expected.dropna()).all()


def test_numeric_looking_text_stays_text(raw: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "numeric_title.csv"
    raw.assign(title="1942").to_csv(path, index=False)
    assert (load_raw(path)["title"] == "1942").all()


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"data/README\.md"):
        load_raw(tmp_path / "absent.csv")


def test_missing_column_raises(raw: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="genre"):
        coerce_raw(raw.drop(columns=["genre"]))


def test_unexpected_columns_are_dropped(raw: pd.DataFrame) -> None:
    frame, report = coerce_raw(raw.assign(extra=1))
    assert "extra" not in frame.columns
    assert report.dropped_columns == ("extra",)


def test_unparseable_values_raise_when_strict(raw: pd.DataFrame) -> None:
    bad = raw.astype({"total_sales": object, "release_date": object})
    bad.loc[0, "total_sales"] = "n/a"
    bad.loc[1, "release_date"] = "not a date"
    with pytest.raises(SchemaError, match="unparseable"):
        coerce_raw(bad)


def test_unparseable_values_are_counted_when_lenient(raw: pd.DataFrame) -> None:
    bad = raw.astype({"total_sales": object, "release_date": object})
    bad.loc[0, "total_sales"] = "n/a"
    bad.loc[1, "release_date"] = "not a date"
    frame, report = coerce_raw(bad, strict=False)
    assert report.unparseable == {"total_sales": 1, "release_date": 1}
    assert np.isnan(frame.loc[0, "total_sales"])
    assert pd.isna(frame.loc[1, "release_date"])


def test_out_of_range_values(raw: pd.DataFrame) -> None:
    bad = raw.copy()
    bad.loc[0, "critic_score"] = 42.0
    with pytest.raises(SchemaError, match="critic_score"):
        coerce_raw(bad)
    frame, report = coerce_raw(bad, strict=False)
    assert report.out_of_range == {"critic_score": 1}
    assert np.isnan(frame.loc[0, "critic_score"])


def test_input_frame_is_not_modified(raw: pd.DataFrame) -> None:
    before = raw.copy()
    coerce_raw(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_already_parsed_frame_is_accepted(csv_path: Path) -> None:
    frame = load_raw(csv_path)
    again, report = coerce_raw(frame)
    pd.testing.assert_frame_equal(again, frame)
    assert report.unparseable == {}
