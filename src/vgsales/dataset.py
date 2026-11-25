"""Modelling dataset: from the raw extract to features, target and holdout.

:func:`prepare_dataset` chains the data steps in their only valid order:

1. Coerce the raw frame to the schema.
2. Clean it, keeping rows with an observed target only.
3. Restrict it to the configured release years and flag right-censoring.
4. Build the pre-release features. Platform launch dates and the platforms
   of each title are estimated from every loaded release, tracked or not,
   since a release on a platform is known whether or not its sales were
   tracked.
5. Split off the holdout of the latest release years, purging development
   rows whose title also appears in the holdout.

No statistic of the target is computed here. Everything learned from the
target happens inside the estimators, fold by fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vgsales.config import Config
from vgsales.data.cleaning import CleaningReport, clean, normalise_text
from vgsales.data.loading import coerce_raw, load_raw
from vgsales.data.series import N_TITLES, aggregate_series, to_wide, uncensored_end
from vgsales.data.splits import holdout_split
from vgsales.data.temporal import (
    CENSORED,
    RELEASE_YEAR,
    TemporalReport,
    apply_release_scope,
)
from vgsales.features.prerelease import GROUP_COLUMN, build_prerelease_features


@dataclass(frozen=True)
class Dataset:
    """Features, target and holdout indices of the modelling rows.

    Attributes
    ----------
    features : pandas.DataFrame
        Output of :func:`vgsales.features.prerelease.build_prerelease_features`
        for every in-scope row with an observed target, including the
        ``title_key`` group label.
    target : numpy.ndarray
        Observed sales on the original scale, aligned with ``features``.
    censored : numpy.ndarray
        Right-censoring flag of each row.
    dev_idx, test_idx : numpy.ndarray
        Positional indices of the development rows and the holdout rows.
    cleaning : CleaningReport
        Row counts of the cleaning step.
    scope : TemporalReport
        Snapshot date and row counts of the release-year filter.
    """

    features: pd.DataFrame
    target: np.ndarray
    censored: np.ndarray
    dev_idx: np.ndarray
    test_idx: np.ndarray
    cleaning: CleaningReport
    scope: TemporalReport

    @property
    def groups(self) -> pd.Series:
        """Title key of each row."""
        return self.features[GROUP_COLUMN]

    def development(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the features and target of the development rows."""
        return self.features.iloc[self.dev_idx], self.target[self.dev_idx]

    def holdout(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the features and target of the holdout rows."""
        return self.features.iloc[self.test_idx], self.target[self.test_idx]

    def summary(self) -> dict[str, Any]:
        """Return sizes and year ranges of the development and holdout sets."""
        years = self.features[RELEASE_YEAR].to_numpy()

        def describe(idx: np.ndarray) -> dict[str, int]:
            return {
                "n_rows": int(idx.size),
                "first_year": int(years[idx].min()),
                "last_year": int(years[idx].max()),
                "n_censored": int(self.censored[idx].sum()),
            }

        return {
            "snapshot": self.scope.snapshot,
            "n_rows": len(self.features),
            "n_purged": len(self.features) - self.dev_idx.size - self.test_idx.size,
            "development": describe(self.dev_idx),
            "holdout": describe(self.test_idx),
        }


def _release_reference(loaded: pd.DataFrame) -> pd.DataFrame:
    reference = loaded.loc[:, ["title", "console", "release_date"]].copy()
    for column in ("title", "console"):
        reference[column] = normalise_text(reference[column])
    return reference


def prepare_dataset(
    raw: pd.DataFrame, config: Config, *, strict: bool = True
) -> Dataset:
    """Turn a raw or loaded extract into the modelling dataset.

    Parameters
    ----------
    raw : pandas.DataFrame
        Frame with the raw schema columns, as read from the CSV or produced
        by :func:`vgsales.data.synthetic.make_synthetic`.
    config : Config
        Pipeline configuration; the ``data``, ``target`` and ``split``
        sections are used.
    strict : bool, default True
        Passed to the schema coercion.

    Returns
    -------
    Dataset
        Features, target, censoring flags and holdout indices.

    Raises
    ------
    ValueError
        If no row survives cleaning and scoping, or the holdout leaves no
        development rows.
    """
    loaded, _ = coerce_raw(raw, strict=strict)
    cleaned, cleaning = clean(loaded, strict=strict)
    scoped, scope = apply_release_scope(cleaned, config.data)
    if scoped.empty:
        raise ValueError("no rows with observed sales in the configured scope")
    features = build_prerelease_features(scoped, reference=_release_reference(loaded))
    target = scoped[config.target.column].to_numpy(dtype=np.float64)
    dev_idx, test_idx = holdout_split(
        features[RELEASE_YEAR],
        config.split.holdout_years,
        groups=features[GROUP_COLUMN],
    )
    return Dataset(
        features=features,
        target=target,
        censored=scoped[CENSORED].to_numpy(dtype=bool),
        dev_idx=dev_idx,
        test_idx=test_idx,
        cleaning=cleaning,
        scope=scope,
    )


def load_dataset(
    config: Config, path: str | Path | None = None, *, strict: bool = True
) -> Dataset:
    """Read the extract from disk and prepare the modelling dataset.

    Parameters
    ----------
    config : Config
        Pipeline configuration.
    path : str or Path, optional
        CSV location. Defaults to ``config.data.raw_path``.
    strict : bool, default True
        Passed to the loader and the schema coercion.

    Returns
    -------
    Dataset
        See :func:`prepare_dataset`.
    """
    raw = load_raw(config.data.raw_path if path is None else path, strict=strict)
    return prepare_dataset(raw, config, strict=strict)


@dataclass(frozen=True)
class SeriesData:
    """Monthly aggregate series for the rolling-origin evaluation.

    Attributes
    ----------
    sales, counts : pandas.DataFrame
        Monthly sales in millions of units and numbers of tracked titles,
        one column per segment (:func:`vgsales.data.series.to_wide`).
    snapshot : str
        Snapshot date of the extract in ISO format.
    """

    sales: pd.DataFrame
    counts: pd.DataFrame
    snapshot: str

    def summary(self) -> dict[str, Any]:
        """Return the range, number of segments and number of titles."""
        return {
            "snapshot": self.snapshot,
            "first_month": self.sales.index[0].strftime("%Y-%m"),
            "last_month": self.sales.index[-1].strftime("%Y-%m"),
            "n_months": len(self.sales),
            "n_segments": self.sales.shape[1],
            "n_titles": int(self.counts.to_numpy().sum()),
        }


def prepare_series(
    raw: pd.DataFrame, config: Config, *, strict: bool = True
) -> SeriesData:
    """Aggregate a raw or loaded extract into monthly segment series.

    The rows pass the same coercion, cleaning and release-year scope as in
    :func:`prepare_dataset`. The series run from ``series.start`` to the
    last month before the censoring window of ``data.censoring_window_years``
    (:func:`vgsales.data.series.uncensored_end`), so that no month is scored
    while its sales are still accumulating.

    Parameters
    ----------
    raw : pandas.DataFrame
        Frame with the raw schema columns.
    config : Config
        Pipeline configuration; the ``data`` and ``series`` sections are used.
    strict : bool, default True
        Passed to the schema coercion.

    Returns
    -------
    SeriesData
        Wide monthly sales and title counts.
    """
    loaded, _ = coerce_raw(raw, strict=strict)
    cleaned, _ = clean(loaded, strict=strict)
    scoped, scope = apply_release_scope(cleaned, config.data)
    if scoped.empty:
        raise ValueError("no rows with observed sales in the configured scope")
    end = uncensored_end(scope.snapshot, config.data.censoring_window_years)
    series = aggregate_series(
        scoped,
        by=config.series.by,
        start=config.series.start,
        end=end,
        min_titles=config.series.min_titles,
    )
    return SeriesData(
        sales=to_wide(series),
        counts=to_wide(series, value=N_TITLES),
        snapshot=scope.snapshot,
    )
