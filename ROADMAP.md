# Roadmap

This file tracks the conversion of the original exploratory notebook into a
reproducible forecasting project. The task is defined as pre-release
forecasting: predict the lifetime global unit sales of a title from attributes
that are known at or before its release date, and validate the predictions on
release years that the model has not seen.

## Target layout

```
.
|-- .github/workflows/ci.yml   lint and tests on every push
|-- configs/                   YAML configuration (data, split, features, models)
|-- data/README.md             source, license, schema, acquisition
|-- docs/                      methodology and model card
|-- notebooks/                 exploration and walkthrough notebooks
|-- reports/figures/           generated figures
|-- src/vgsales/
|   |-- data/                  schema, loading, cleaning, splits, synthetic data
|   |-- features/              leakage-safe encoders, pre-release features
|   |-- models/                baselines, gradient boosting, PyTorch models
|   |-- tuning/                Optuna search with time-series CV
|   |-- evaluation/            metrics, cross-validation runner, reports
|   |-- explain/               SHAP and permutation importance
|   |-- monitoring/            drift statistics and reports
|   `-- serving/               model bundle and FastAPI application
|-- tests/                     unit and integration tests on synthetic data
|-- Dockerfile
|-- LICENSE
|-- Makefile
`-- pyproject.toml
```

## Known issues

Findings from the audit of `ml-models-and-cross-validation.ipynb` and
`README.md`. Each is addressed by one or more items below.

1. Integer cast destroys features and target. `encoded_df.astype(int)`
   truncates the frequency encodings of `console`, `publisher` and `developer`
   to zero (standard deviation 0.0 in the notebook output), truncates
   `total_sales` to whole millions (mean falls from 0.116 to 0.049) and
   truncates `critic_score`. All reported metrics refer to an integer target
   and constant categorical features.
2. Target leakage through regional sales. `na_sales`, `jp_sales`, `pal_sales`
   and `other_sales` are used as features, while `total_sales` is their sum up
   to rounding. The model reconstructs the target from its components, and
   none of these columns exists before release.
3. Missing target treated as zero. 45,094 of 64,016 rows (70.4%) have no
   recorded `total_sales`. Filling them with 0 turns "not tracked" into "zero
   sales" and dominates the target distribution.
4. Target statistics computed on all rows. `publisher_avg_sales` and
   `developer_avg_sales` are group means of the target over the full dataset,
   test rows included. After issue 1 they collapse to a single constant.
5. Encoders fitted before the split. Frequency encoding and the per-console
   critic score normalisation use the full dataset.
6. Index misalignment during cleaning. `critic_score` imputation and date
   parsing read from the uncleaned frame `df` rather than `df_cleaned`. The
   `last_update` fill is overwritten by re-parsing `df['last_update']`,
   leaving 44,376 missing values that are later cast to integers.
7. Splits ignore time and title identity. The 80/20 split and all
   cross-validation schemes shuffle rows, so later releases inform
   predictions for earlier ones, and a title released on several platforms
   appears on both sides of the split.
8. Cross-validation is misread. The unshuffled `KFold` fold with RMSE 0.965
   reflects rows ordered by sales (the first fold holds the best sellers). The
   gain under `StratifiedKFold` comes from shuffling, and stratifying a
   continuous target only runs because of the integer cast (scikit-learn warns
   that a class has one member).
9. Model selection on the test set. The "tuned" XGBoost configuration was
   chosen by comparing test-set scores.
10. The serialized model is unfitted. `joblib.dump` saves the estimator that
    was passed to `cross_val_score`, which fits clones, so
    `sales_forecasting_xgb.pkl` holds an unfitted model. The file is also not
    in the repository.
11. Metrics on the raw scale only. For a heavy-tailed target, RMSE on raw
    units is dominated by a few titles. There are no log-scale metrics and no
    uncertainty estimates.
12. Documentation errors. The README mentions `nameOrig` and `nameDest`,
    which are not columns of this dataset, lists label encoding, which is not
    used, lists a `.pkl` file that is absent, and gives a clone URL that does
    not match the repository name.
13. No reproducibility infrastructure: no dependency specification, tests,
    configuration, data documentation or license, and hard-coded Kaggle
    paths.

## Phase 1: Foundation


## Phase 2: Data pipeline and time-aware validation

- [ ] 2.2 Implement cleaning: keep rows with an observed target only (no zero imputation), parse dates, drop exact duplicates, normalise categorical strings.
- [ ] 2.3 Add release-year filtering and a right-censoring flag for titles released close to the data snapshot.
- [ ] 2.4 Implement a release-year holdout split and an expanding-window cross-validator over release years.
- [ ] 2.5 Group titles across platforms so that a title never appears on both sides of a split.
- [ ] 2.6 Implement a leakage-safe frequency encoder fitted on training folds only.
- [ ] 2.7 Implement a temporal target encoder for publisher and developer that uses only strictly earlier release years, with smoothing.
- [ ] 2.8 Implement the pre-release feature builder (calendar features, platform lifecycle stage, platform count, critic score with a missingness indicator), excluding regional sales by construction.
- [ ] 2.9 Assemble the scikit-learn preprocessing pipeline and add leakage tests (future rows do not change past features, regional columns are rejected).
- [ ] 2.10 Move the original notebook to `notebooks/00_original_exploration.ipynb` with a header cell that lists the known issues.
- [ ] 2.11 Add `notebooks/01_eda.ipynb` built on the package: target distribution, coverage by release year, missingness, censoring.

## Phase 3: Evaluation framework and baselines

- [ ] 3.1 Implement metrics on the log and original scales (RMSLE, MAE, median absolute error, R^2) with tests.
- [ ] 3.2 Implement the cross-validation runner that returns per-fold and aggregate metric tables.
- [ ] 3.3 Add baselines: global median, genre-by-platform median, and ridge regression on one-hot features.
- [ ] 3.4 Add a command-line entry point (`vgsales train`, `vgsales evaluate`) driven by the configuration.
- [ ] 3.5 Add a report writer for JSON metrics and markdown comparison tables under `reports/`.

## Phase 4: Gradient boosting

- [ ] 4.1 Add an XGBoost regressor with early stopping on a temporal validation fold.
- [ ] 4.2 Add a LightGBM regressor with native categorical handling.
- [ ] 4.3 Add a CatBoost regressor with ordered target statistics.
- [ ] 4.4 Add a model registry and compare baselines and boosting models on identical folds.
- [ ] 4.5 Add a paired Wilcoxon signed-rank test on per-fold errors to the comparison report.

## Phase 5: Deep learning (PyTorch)

- [ ] 5.1 Add a tabular dataset and data loader for categorical indices and standardised numeric inputs.
- [ ] 5.2 Implement an entity-embedding MLP with dropout and weight decay.
- [ ] 5.3 Implement the training loop with early stopping, learning-rate scheduling and seeded determinism.
- [ ] 5.4 Register the MLP in the model registry and the cross-validation runner.
- [ ] 5.5 Build aggregate sales series (sales by release month for each genre and platform family).
- [ ] 5.6 Implement a GRU sequence model for the aggregate series with windowed training.
- [ ] 5.7 Evaluate the sequence model with rolling-origin validation against seasonal naive and exponential smoothing baselines.

## Phase 6: Hyperparameter optimisation

- [ ] 6.1 Define Optuna search spaces for XGBoost, LightGBM and CatBoost.
- [ ] 6.2 Implement the Optuna objective on the expanding-window cross-validator with median pruning.
- [ ] 6.3 Persist best parameters to `configs/tuned/` and add a `vgsales tune` command.
- [ ] 6.4 Add an Optuna search for the embedding MLP.

## Phase 7: Prediction intervals

- [ ] 7.1 Add LightGBM quantile regression (pinball loss) and interval metrics: coverage, mean width, interval score.
- [ ] 7.2 Add split conformal prediction with a calibration set drawn from later release years.
- [ ] 7.3 Add conformalized quantile regression.
- [ ] 7.4 Evaluate coverage by release year and genre, with a reliability figure.

## Phase 8: Explainability

- [ ] 8.1 Add SHAP TreeExplainer global importance and summary figures.
- [ ] 8.2 Add SHAP dependence plots and local explanations for individual titles.
- [ ] 8.3 Compute permutation importance on the temporal holdout and compare with the SHAP ranking.

## Phase 9: Experiment tracking

- [ ] 9.1 Add MLflow tracking utilities behind an optional-dependency guard.
- [ ] 9.2 Log cross-validation, tuning and final training runs, and register the selected model.

## Phase 10: Serving

- [ ] 10.1 Add a model bundle (fitted pipeline, conformal quantile, feature metadata, version) with save and load.
- [ ] 10.2 Add a FastAPI service with pydantic schemas, `/health` and `/predict` (point estimate and interval), and API tests.
- [ ] 10.3 Add a batch prediction endpoint and structured validation errors.
- [ ] 10.4 Add a slim, non-root `Dockerfile` and Makefile targets to build and run the image.

## Phase 11: Drift monitoring

- [ ] 11.1 Implement population stability index, Kolmogorov-Smirnov and chi-square drift statistics with tests.
- [ ] 11.2 Build a drift report that compares a reference window with a new batch, including prediction drift.
- [ ] 11.3 Add a `vgsales drift` command and an example report under `reports/`.

## Phase 12: Documentation

- [ ] 12.1 Write `docs/methodology.md`: problem statement, notation, validation design, models, intervals, references.
- [ ] 12.2 Write `docs/model_card.md` following Mitchell et al. (2019).
- [ ] 12.3 Add `notebooks/02_modeling.ipynb`, an end-to-end walkthrough built on the package.
- [ ] 12.4 Rewrite the README: motivation, results from full runs, reproduction steps, project layout.

## Completed

- [x] 2026-11-19: 1.1 Add `pyproject.toml` (src layout, Python 3.10+, core dependencies, optional extras for heavy libraries), `LICENSE`, `.gitignore`, package skeleton and a smoke test.
- [x] 2026-11-19: 1.2 Add typed configuration: `configs/default.yaml` and a validated dataclass loader in `src/vgsales/config.py`.
- [x] 2026-11-19: 1.3 Add the raw data schema (column groups, dtypes, leakage list) in `src/vgsales/data/schema.py`.
- [x] 2026-11-19: 1.4 Add a seeded synthetic generator that reproduces the VGChartz 2024 schema and missingness pattern for tests.
- [x] 2026-11-19: 1.5 Add a `Makefile` (install, lint, format, test) and a GitHub Actions workflow running ruff and pytest on Python 3.10 and 3.12.
- [x] 2026-11-19: 1.6 Add `data/README.md` (source, license, schema, acquisition, coverage limits) and the `reports/figures/` directory.
- [x] 2026-11-20: 2.1 Implement the CSV loader with schema validation and dtype coercion.
