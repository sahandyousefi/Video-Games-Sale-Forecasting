"""Seeded synthetic data with the layout of the VGChartz 2024 extract.

The test suite and CI use this generator because the real file cannot be
redistributed. It reproduces the column layout, the raw CSV representation
(dates as ISO strings), the multi-platform structure of titles, the dependence
of tracking on title size, and the high missingness of sales and critic
scores. Effect sizes are arbitrary, so metrics computed on this data carry no
information about the real market.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from vgsales.data.schema import RAW_COLUMNS

# First and last release year of titles on each platform.
PLATFORM_WINDOWS: dict[str, tuple[int, int]] = {
    "PS": (1995, 2003),
    "PS2": (2000, 2012),
    "PS3": (2006, 2017),
    "PS4": (2013, 2022),
    "PS5": (2020, 2024),
    "X360": (2005, 2016),
    "XOne": (2013, 2021),
    "Wii": (2006, 2015),
    "NS": (2017, 2024),
    "GBA": (2001, 2008),
    "DS": (2004, 2014),
    "3DS": (2011, 2019),
    "PSP": (2004, 2014),
    "PC": (1985, 2024),
}

GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Fighting",
    "Misc",
    "Platform",
    "Puzzle",
    "Racing",
    "Role-Playing",
    "Shooter",
    "Simulation",
    "Sports",
    "Strategy",
)

# Dirichlet concentration for the NA, JP, PAL and other shares of total sales.
_REGION_ALPHA = np.array([4.0, 1.5, 3.0, 1.0])
_MIN_YEAR = min(a for a, _ in PLATFORM_WINDOWS.values())
_MAX_YEAR = max(b for _, b in PLATFORM_WINDOWS.values())


@dataclass(frozen=True)
class SyntheticSpec:
    """Size and missingness parameters of the synthetic extract."""

    n_titles: int = 400
    start_year: int = 1998
    end_year: int = 2022
    n_publishers: int = 30
    n_developers: int = 80
    p_sales_observed: float = 0.35
    p_critic_observed: float = 0.15
    p_release_missing: float = 0.10
    p_developer_missing: float = 0.005
    p_region_missing: float = 0.15
    snapshot: str = "2024-06-30"

    def __post_init__(self) -> None:
        if self.n_titles < 1 or self.n_publishers < 1 or self.n_developers < 1:
            raise ValueError("counts must be positive")
        if not _MIN_YEAR <= self.start_year <= self.end_year <= _MAX_YEAR:
            raise ValueError(
                f"years must satisfy {_MIN_YEAR} <= start_year <= end_year "
                f"<= {_MAX_YEAR}"
            )
        probabilities = (
            self.p_sales_observed,
            self.p_critic_observed,
            self.p_release_missing,
            self.p_developer_missing,
            self.p_region_missing,
        )
        if not all(0.0 <= p <= 1.0 for p in probabilities):
            raise ValueError("probabilities must lie in [0, 1]")
        date.fromisoformat(self.snapshot)


def _zipf_weights(n: int) -> np.ndarray:
    weights = 1.0 / np.arange(1, n + 1)
    return weights / weights.sum()


def _random_date(rng: np.random.Generator, year: int) -> date:
    return date(year, 1, 1) + timedelta(days=int(rng.integers(0, 365)))


def make_synthetic(spec: SyntheticSpec | None = None, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic frame in the raw VGChartz layout.

    Log sales follow an additive model in publisher, developer, genre and
    platform effects, a latent title quality and a platform lifecycle term,
    with Gaussian noise. Larger titles are more likely to have tracked sales,
    which mimics the selection in the real data.

    Parameters
    ----------
    spec : SyntheticSpec, optional
        Generator parameters. Defaults to ``SyntheticSpec()``.
    seed : int, default 0
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    pandas.DataFrame
        One row per (title, platform) pair with the columns of
        :data:`vgsales.data.schema.RAW_COLUMNS`. Dates are ISO strings and
        missing values are NaN, as after ``pandas.read_csv``.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(seed)
    snapshot = date.fromisoformat(spec.snapshot)

    publishers = [f"Publisher {i:02d}" for i in range(spec.n_publishers)]
    publisher_weights = _zipf_weights(spec.n_publishers)
    publisher_effect = rng.normal(0.0, 0.8, spec.n_publishers)
    developer_owner = rng.integers(0, spec.n_publishers, spec.n_developers)
    developer_effect = rng.normal(0.0, 0.5, spec.n_developers)
    genre_effect = dict(zip(GENRES, rng.normal(0.0, 0.4, len(GENRES)), strict=True))
    platform_effect = dict(
        zip(
            PLATFORM_WINDOWS,
            rng.normal(0.0, 0.5, len(PLATFORM_WINDOWS)),
            strict=True,
        )
    )

    rows: list[dict[str, object]] = []
    for t in range(spec.n_titles):
        year = int(rng.integers(spec.start_year, spec.end_year + 1))
        available = [p for p, (a, b) in PLATFORM_WINDOWS.items() if a <= year <= b]
        n_platforms = min(len(available), 1 + int(rng.poisson(0.5)))
        platforms = rng.choice(available, size=n_platforms, replace=False)

        pub = int(rng.choice(spec.n_publishers, p=publisher_weights))
        owned = np.flatnonzero(developer_owner == pub)
        if owned.size:
            dev = int(rng.choice(owned))
        else:
            dev = int(rng.integers(0, spec.n_developers))
        genre = str(rng.choice(GENRES))
        quality = float(rng.normal())

        for platform in platforms:
            first, last = PLATFORM_WINDOWS[str(platform)]
            lifecycle = (year - first) / max(last - first, 1)
            mu = (
                -2.5
                + publisher_effect[pub]
                + developer_effect[dev]
                + genre_effect[genre]
                + platform_effect[str(platform)]
                + 0.5 * quality
                - 0.8 * lifecycle
            )
            sales = max(round(float(np.exp(mu + rng.normal(0.0, 0.8))), 2), 0.01)
            # Tracking probability rises with quality, which also raises sales.
            p_observed = np.clip(
                spec.p_sales_observed * (1 + 0.5 * quality), 0.02, 0.98
            )
            observed = rng.random() < p_observed

            release = _random_date(rng, year)
            row: dict[str, object] = {
                "img": f"/games/boxart/{t:05d}_{platform}.jpg",
                "title": f"Title {t:04d}",
                "console": str(platform),
                "genre": genre,
                "publisher": publishers[pub],
                "developer": f"Studio {dev:03d}",
                "critic_score": np.nan,
                "total_sales": np.nan,
                "na_sales": np.nan,
                "jp_sales": np.nan,
                "pal_sales": np.nan,
                "other_sales": np.nan,
                "release_date": release.isoformat(),
                "last_update": np.nan,
            }
            if rng.random() < spec.p_critic_observed:
                score = 7.0 + 0.8 * quality + rng.normal(0.0, 0.6)
                row["critic_score"] = round(float(np.clip(score, 1.0, 10.0)), 1)
            if observed:
                row["total_sales"] = sales
                shares = rng.dirichlet(_REGION_ALPHA)
                for col, share in zip(
                    ("na_sales", "jp_sales", "pal_sales", "other_sales"),
                    shares,
                    strict=True,
                ):
                    value = round(sales * float(share), 2)
                    if value > 0 and rng.random() >= spec.p_region_missing:
                        row[col] = value
                if rng.random() < 0.95 and release < snapshot:
                    span = (snapshot - release).days
                    updated = release + timedelta(days=int(rng.integers(0, span + 1)))
                    row["last_update"] = updated.isoformat()
            if rng.random() < spec.p_release_missing:
                row["release_date"] = np.nan
            if rng.random() < spec.p_developer_missing:
                row["developer"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))
