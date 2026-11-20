"""Row-level cleaning of the raw extract.

Cleaning keeps only rows with an observed target. A missing ``total_sales``
means the title was not tracked by VGChartz, so it is excluded rather than
imputed with zero (known issue 3 in ``ROADMAP.md``). No other values are
imputed at this stage; imputation of inputs belongs to the fitted feature
pipeline.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from vgsales.data.loading import coerce_raw
from vgsales.data.schema import CATEGORICAL, IDENTIFIERS, TARGET

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

# Duplicate (title, console) keys with conflicting values are reported, not
# resolved, because the extract gives no basis for choosing between them.
KEY_COLUMNS = ("title", "console")


@dataclass(frozen=True)
class CleaningReport:
    """Row counts removed by each cleaning step.

    Attributes
    ----------
    n_input : int
        Rows received.
    n_untracked : int
        Rows dropped because ``total_sales`` is missing.
    n_duplicates : int
        Exact duplicate rows dropped after text normalisation.
    n_output : int
        Rows returned.
    n_conflicting_keys : int
        Rows in the output that share a ``(title, console)`` key with another
        row. These are kept and only reported.
    """

    n_input: int
    n_untracked: int
    n_duplicates: int
    n_output: int
    n_conflicting_keys: int

    def to_dict(self) -> dict[str, int]:
        """Return the counts as a plain dictionary."""
        return asdict(self)


def _normalise_value(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return _WHITESPACE.sub(" ", text).strip()


def normalise_text(series: pd.Series) -> pd.Series:
    """Normalise free-text values for consistent grouping.

    Applies Unicode NFKC normalisation (which maps, for example, non-breaking
    and full-width spaces to their plain forms), collapses runs of whitespace
    and strips both ends. Values that become empty are set to missing. Case
    is preserved because publisher and developer names are case-sensitive
    labels in the source.

    Parameters
    ----------
    series : pandas.Series
        Text values with ``NaN`` for missing entries.

    Returns
    -------
    pandas.Series
        Object-dtype series with normalised text and ``NaN`` for missing.
    """
    out = series.astype(object)
    present = out.notna()
    out[present] = [_normalise_value(str(v)) for v in out[present]]
    return out.where(present & out.ne(""), np.nan)


def clean(
    df: pd.DataFrame, *, strict: bool = True
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw frame for modelling.

    Steps, in order:

    1. Coerce to the schema dtypes with
       :func:`vgsales.data.loading.coerce_raw`, which parses the date columns.
    2. Normalise identifier and categorical text with :func:`normalise_text`.
    3. Drop rows without an observed target.
    4. Drop exact duplicate rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw or loaded frame with the schema columns.
    strict : bool, default True
        Passed to :func:`vgsales.data.loading.coerce_raw`.

    Returns
    -------
    frame : pandas.DataFrame
        Cleaned frame with a fresh ``RangeIndex``.
    report : CleaningReport
        Row counts for each step.
    """
    frame, _ = coerce_raw(df, strict=strict)
    n_input = len(frame)

    for col in (*IDENTIFIERS, *CATEGORICAL):
        frame[col] = normalise_text(frame[col])

    tracked = frame[TARGET].notna()
    frame = frame.loc[tracked]
    n_untracked = int((~tracked).sum())

    duplicated = frame.duplicated(keep="first")
    frame = frame.loc[~duplicated].reset_index(drop=True)
    n_duplicates = int(duplicated.sum())

    conflicting = frame.duplicated(subset=list(KEY_COLUMNS), keep=False)
    report = CleaningReport(
        n_input=n_input,
        n_untracked=n_untracked,
        n_duplicates=n_duplicates,
        n_output=len(frame),
        n_conflicting_keys=int(conflicting.sum()),
    )
    logger.info("cleaning: %s", report.to_dict())
    return frame, report
