# Data

The raw file is not versioned in this repository. This document describes its
source, layout and the properties that constrain the modelling choices.

## Source

- Dataset: "Video Game Sales 2024" on Kaggle (`asaniczka/video-game-sales-2024`),
  DOI [10.34740/KAGGLE/DSV/7507070](https://doi.org/10.34740/KAGGLE/DSV/7507070).
- Underlying source: sales estimates published by VGChartz
  ([vgchartz.com](https://www.vgchartz.com)), collected by web scraping.
- File: `vgchartz-2024.csv`, 64,016 rows and 14 columns.
- Terms of use: see the Kaggle dataset page. The file is not redistributed here.

## Acquisition

With the Kaggle CLI configured:

```bash
kaggle datasets download -d asaniczka/video-game-sales-2024 -p data/raw --unzip
```

The pipeline expects `data/raw/vgchartz-2024.csv`, the value of
`data.raw_path` in `configs/default.yaml`.

`vgsales.data.loading.load_raw` reads every column as text and then coerces
it to the schema: text columns to `object` with `NaN` for missing values,
numeric columns to `float64` and dates to `datetime64[ns]`. By default an
unparseable value or a value outside its schema range raises `SchemaError`;
with `strict=False` such values are set to missing and counted in a
`LoadReport`. The loader removes no rows.

`vgsales.data.cleaning.clean` then normalises text (Unicode NFKC, collapsed
whitespace, empty strings to missing), drops rows without `total_sales` and
drops exact duplicates. Rows that share a `(title, console)` key but differ in
other values are kept and counted in the `CleaningReport`, since the extract
gives no basis for choosing between them.

## Unit of observation

One row per (title, console) pair. A title released on several platforms
appears once per platform, so rows are not independent: the splitting code
keeps all rows of a title on the same side of every split.

## Schema

| Column | Type | Description | Role |
|---|---|---|---|
| `img` | string | Relative URL of the box art | identifier, dropped |
| `title` | string | Game title | grouping key, not an input |
| `console` | string | Platform code (for example `PS4`, `NS`, `PC`) | input |
| `genre` | string | Genre label | input |
| `publisher` | string | Publisher name | input |
| `developer` | string | Developer name | input |
| `critic_score` | float | Aggregated critic score on a 0 to 10 scale | input |
| `total_sales` | float | Global units sold, millions | target |
| `na_sales` | float | North American units, millions | excluded (target component) |
| `jp_sales` | float | Japanese units, millions | excluded (target component) |
| `pal_sales` | float | European and PAL-region units, millions | excluded (target component) |
| `other_sales` | float | Rest-of-world units, millions | excluded (target component) |
| `release_date` | date | Release date on the platform | input (calendar features, splitting) |
| `last_update` | date | Date the sales record was last updated | excluded (post-release) |

The column groups and the leakage list are defined in
`src/vgsales/data/schema.py`.

## Coverage

Non-missing counts in the full file:

| Column | Non-missing | Share |
|---|---:|---:|
| `total_sales` | 18,922 | 29.6% |
| `na_sales` | 12,637 | 19.7% |
| `jp_sales` | 6,726 | 10.5% |
| `pal_sales` | 12,824 | 20.0% |
| `other_sales` | 15,128 | 23.6% |
| `critic_score` | 6,678 | 10.4% |
| `release_date` | 56,965 | 89.0% |
| `last_update` | 17,879 | 27.9% |
| `developer` | 63,999 | 100.0% |

## Limitations

1. Missing sales mean "not tracked", not "zero units". For this reason, rows
   without `total_sales` are not used for training or evaluation and are not
   imputed. Every model is therefore conditional on a title being tracked.
2. Tracking is not random. Larger releases are more likely to have recorded
   sales, so the labelled sample over-represents commercially visible titles.
3. Sales are third-party estimates, not audited publisher figures.
4. Recent releases are right-censored: their lifetime sales were still
   accumulating at the snapshot date. Titles released within
   `data.censoring_window_years` of the snapshot are flagged in the
   `censored` column by `vgsales.data.temporal.apply_release_scope` and
   reported separately in evaluation. The snapshot is `data.snapshot_date`
   when set, otherwise the latest `last_update` in the file; this is the only
   use of `last_update`. Rows without a release date, outside the configured
   release years or released after the snapshot are removed at the same
   step.
5. `critic_score` is missing for about 90% of rows and is only partly
   available before release. It enters the feature set together with a
   missingness indicator (roadmap item 2.8).

## Synthetic data

`vgsales.data.synthetic.make_synthetic` generates frames with the same columns,
raw representation and missingness pattern. The test suite and CI run on
this data. Metrics computed on synthetic data are labelled as synthetic and are
never reported as results for the real dataset.
