import numpy as np
import pandas as pd
import pytest

from vgsales.data import schema


def _frame() -> pd.DataFrame:
    df = pd.DataFrame({col: [None, None] for col in schema.RAW_COLUMNS})
    for col in (*schema.IDENTIFIERS, *schema.CATEGORICAL):
        df[col] = ["a", "b"]
    for col in schema.NUMERIC:
        df[col] = [1.0, np.nan]
    for col in schema.DATES:
        df[col] = pd.to_datetime(["2010-01-01", None])
    return df


def test_column_groups_partition_raw_columns() -> None:
    groups = [schema.IDENTIFIERS, schema.CATEGORICAL, schema.NUMERIC, schema.DATES]
    flat = [col for group in groups for col in group]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(schema.RAW_COLUMNS)


def test_regional_sales_are_leakage() -> None:
    assert set(schema.REGIONAL_SALES) <= set(schema.LEAKAGE_COLUMNS)


def test_check_columns_reports_missing_and_unexpected() -> None:
    df = _frame().drop(columns=["genre"]).assign(extra=1)
    report = schema.check_columns(df)
    assert report.missing == ("genre",)
    assert report.unexpected == ("extra",)
    assert not report.ok


def test_check_columns_accepts_full_schema() -> None:
    assert schema.check_columns(_frame()).ok


def test_check_dtypes_flags_wrong_kinds() -> None:
    df = _frame()
    assert schema.check_dtypes(df) == {}
    df["release_date"] = ["2010-01-01", None]
    df["critic_score"] = ["8.0", None]
    assert set(schema.check_dtypes(df)) == {"release_date", "critic_score"}


def test_check_ranges_counts_violations() -> None:
    df = pd.DataFrame(
        {"critic_score": [5.0, 11.0, np.nan], "total_sales": [-0.1, 0.2, -1.0]}
    )
    assert schema.check_ranges(df) == {"critic_score": 1, "total_sales": 2}


@pytest.mark.parametrize("bad", ["na_sales", "total_sales", "last_update"])
def test_assert_no_leakage_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match=bad):
        schema.assert_no_leakage(["console", bad])


def test_assert_no_leakage_accepts_pre_release_columns() -> None:
    schema.assert_no_leakage([*schema.CATEGORICAL, "critic_score", "release_date"])
