# Video Game Sales Forecasting Using Machine Learning and Cross-Validation

A machine learning project that builds and compares multiple regression models to predict video game sales using the VGChartz 2024 dataset. The pipeline covers data preprocessing, cardinality-aware categorical encoding, domain-driven feature engineering, multi-model training and comparison (Random Forest, XGBoost, LightGBM), three cross-validation strategies, hyperparameter tuning, and model serialization for deployment.

---

## Overview

Accurately forecasting video game sales before or at launch is a high-value capability for publishers, developers, and investors. This project constructs a robust sales prediction system by analyzing game attributes such as platform, publisher, developer, critic scores, genre, and regional sales data. Three gradient boosting and ensemble models are trained and evaluated side by side, with the best-performing model selected for deployment.

The dataset is sourced from the VGChartz 2024 dataset on Kaggle.

---

## Dataset

The dataset (`vgchartz-2024.csv`) contains historical video game sales records across platforms, regions, and genres.

| Feature | Description |
|---|---|
| `title` | Name of the video game (dropped — not used as a predictor) |
| `console` | Gaming platform (e.g., PS5, Xbox, PC) — high cardinality |
| `genre` | Game genre (e.g., Action, Sports, RPG) — low cardinality |
| `publisher` | Publishing company — high cardinality |
| `developer` | Development studio — high cardinality |
| `critic_score` | Aggregated critic review score |
| `total_sales` | Total global sales in millions — primary target variable |
| `na_sales` | North America regional sales |
| `jp_sales` | Japan regional sales |
| `pal_sales` | Europe and PAL territories sales |
| `other_sales` | Rest of world sales |
| `release_date` | Original release date |
| `last_update` | Date of last sales data update |
| `img` | Image URL — dropped as irrelevant |

**Target Variable:** `total_sales` — global sales in millions of units.

---

## Methodology

### 1. Data Preprocessing

- Dropped rows with missing `release_date` or `developer`, as these are critical structural fields with no reliable imputation strategy
- Filled missing values in all five sales columns (`total_sales`, `na_sales`, `jp_sales`, `pal_sales`, `other_sales`) with 0, representing games with no recorded sales in that region
- Filled missing `last_update` with `release_date` where absent
- Imputed missing `critic_score` values with the column mean to preserve rows without sacrificing data
- Confirmed zero duplicate records in the cleaned dataset
- Converted `release_date` and `last_update` to proper `datetime` format using `pd.to_datetime`
- Dropped the `img` column as it carries no predictive information

---

### 2. Categorical Encoding

Categorical variables were encoded using two distinct strategies based on cardinality — the number of unique values per column:

**Frequency Encoding (High Cardinality)**

Applied to `console`, `publisher`, and `developer`. Each category was replaced with its relative frequency in the dataset (proportion of total rows). This preserves information about how common each category is without creating thousands of binary columns.

    freq_map = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq_map)

**One-Hot Encoding (Low Cardinality)**

Applied to `genre` using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity. Each genre became a separate binary column.

---

### 3. Exploratory Data Analysis and Visualization

- **Sales Distribution Histogram:** Total sales are heavily right-skewed — the vast majority of games sell below 2.5 million copies, while a small number of blockbuster titles drive the upper tail. This skewness informed the log transformation applied later.
- **Correlation Heatmap:** Regional sales (`na_sales`, `jp_sales`, `pal_sales`, `other_sales`) are strongly correlated with `total_sales` — as expected, since total is the sum of regional sales. To prevent data leakage, `total_sales` was excluded from the feature set during model training.
- **Critic Score Correlation:** `critic_score` showed only weak positive correlation (~0.1 to 0.3) with sales, suggesting that commercial success is driven more by marketing, platform presence, and franchise recognition than review scores.
- **Average Sales by Genre:** Bar chart visualizing which genres generate the highest average sales across the dataset.

---

### 4. Feature Engineering

The following engineered features were added to improve model signal:

| Feature | Description |
|---|---|
| `publisher_avg_sales` | Mean historical total sales per publisher — proxy for publisher market strength |
| `developer_avg_sales` | Mean historical total sales per developer — proxy for developer track record |
| `critic_score_normalized` | Critic score divided by the maximum score on that console — makes scores comparable across platforms |
| `console_{genre}` interaction terms | Product of console frequency encoding and each genre flag — captures platform-genre affinity |
| `total_sales_log` | Log-transformed target (`log1p`) — computed for analysis but excluded from features to prevent leakage |

---

### 5. Machine Learning Models

All models were trained on an 80/20 train-test split with `random_state=42`. `total_sales` and `total_sales_log` were excluded from the feature matrix to prevent leakage. `nameOrig`, `nameDest`, and `title` were dropped prior to training.

---

#### 5.1 Random Forest Regressor

A Random Forest with 200 estimators was trained as the baseline ensemble model.

**Configuration:**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `random_state` | 42 |

**Results:**

| Metric | Value |
|---|---|
| MAE | 0.0341 |
| RMSE | 0.1552 |
| R² Score | 0.8452 |

**Interpretation:** The Random Forest explains 84.52% of the variance in sales, with low average prediction error. It serves as a strong baseline but leaves room for improvement with gradient boosting methods.

---

#### 5.2 XGBoost Regressor — Baseline

XGBoost was trained with default-tuned parameters and compared directly against Random Forest.

**Configuration:**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.1 |
| `max_depth` | 8 |
| `random_state` | 42 |

**Results:**

| Metric | Value |
|---|---|
| MAE | 0.0341 |
| RMSE | 0.1430 |
| R² Score | 0.8686 |

**Interpretation:** XGBoost outperforms Random Forest with a lower RMSE and higher R², explaining 86.86% of variance. The lower RMSE indicates fewer large prediction errors.

---

#### 5.3 XGBoost Regressor — Hyperparameter Tuned

The XGBoost model was refined with additional regularization and subsampling parameters to reduce overfitting and improve generalization.

**Tuned Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 300 | More trees for better learning |
| `learning_rate` | 0.05 | Slower, more stable updates |
| `max_depth` | 10 | Deeper trees to capture complex patterns |
| `subsample` | 0.8 | 80% of rows per tree — reduces overfitting |
| `colsample_bytree` | 0.8 | 80% of features per tree — reduces overfitting |
| `random_state` | 42 | Reproducibility |

**Results:**

| Metric | Value |
|---|---|
| MAE | 0.0340 |
| RMSE | 0.1427 |
| R² Score | 0.8692 |

**Interpretation:** Tuning improved both RMSE and R² marginally, confirming the parameter choices reduce overfitting and improve stability. This is the final selected model.

---

#### 5.4 LightGBM Regressor

LightGBM was evaluated as a fast alternative to XGBoost, trained with matching hyperparameters for a fair comparison.

**Configuration:** Identical to Tuned XGBoost (`n_estimators=300`, `learning_rate=0.05`, `max_depth=10`, `subsample=0.8`, `colsample_bytree=0.8`)

**Results:**

| Metric | Value |
|---|---|
| MAE | 0.0399 |
| RMSE | 0.1447 |
| R² Score | 0.8655 |

**Interpretation:** LightGBM performs competitively but does not surpass the tuned XGBoost model. Higher MAE and RMSE, combined with warnings about features yielding no positive gain splits, suggest that further feature selection could benefit LightGBM specifically.

---

### 6. Model Comparison Summary

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Random Forest | 0.0341 | 0.1552 | 0.8452 |
| XGBoost (Baseline) | 0.0341 | 0.1430 | 0.8686 |
| XGBoost (Tuned) | 0.0340 | 0.1427 | 0.8692 |
| LightGBM | 0.0399 | 0.1447 | 0.8655 |

**Winner: Tuned XGBoost** — lowest RMSE, highest R², and most consistent generalization across validation strategies.

---

### 7. Cross-Validation

Three cross-validation strategies were applied to the tuned XGBoost model to assess generalization robustness.

**7.1 Standard 5-Fold Cross-Validation**

| Metric | Value |
|---|---|
| RMSE per fold | [0.9652, 0.0432, 0.0669, 0.0707, 0.0833] |
| Mean RMSE | 0.2470 |
| Std Dev | 0.3593 |

One fold produced an anomalously high RMSE of 0.9652, inflating the mean and standard deviation. This indicates sensitivity to data distribution within folds, motivating the switch to stratified splitting.

---

**7.2 Stratified K-Fold Cross-Validation (5 Folds)**

Stratified K-Fold ensures that each fold preserves the distribution of the target variable, producing more balanced and representative splits.

| Metric | Value |
|---|---|
| Mean RMSE | 0.1379 |
| Std Dev | 0.0024 |

The dramatic reduction in standard deviation (from 0.3593 to 0.0024) confirms that the high variance in standard K-Fold was caused by imbalanced target distribution across folds. Stratified splitting resolves this and reveals the model's true generalization performance.

---

**7.3 ShuffleSplit Cross-Validation (5 Splits, 20% Test)**

ShuffleSplit randomly resamples train/test splits without regard to class stratification, providing an independent generalization estimate.

| Metric | Value |
|---|---|
| RMSE per split | [0.1427, 0.1355, 0.1433, 0.1339, 0.1527] |
| Mean RMSE | 0.1416 |
| Std Dev | 0.0067 |

Consistent RMSE values across all five randomized splits confirm stable model behavior. The low standard deviation (0.0067) validates that the model generalizes reliably to unseen data regardless of the specific split.

---

### 8. Feature Importance

XGBoost's built-in feature importance scores were plotted for the top 10 most influential features. Key findings:

- Regional sales features (`na_sales`, `pal_sales`, `jp_sales`, `other_sales`) dominate importance rankings, confirming they are the strongest predictors of total global sales
- `publisher_avg_sales` and `developer_avg_sales` (engineered features) ranked highly, validating the value of historical performance proxies
- `critic_score` and `critic_score_normalized` contributed but were not among the top predictors
- Genre and platform interaction terms provided marginal but measurable signal

---

### 9. Deployment Preparation

The final tuned XGBoost model was serialized using `joblib` for deployment:

    joblib.dump(xgb_model, "sales_forecasting_xgb.pkl")

A `predict_sales` inference function was defined to accept new input data and return sales predictions using the loaded model. An actual vs. predicted sales scatter plot was generated as a final visual validation, confirming that most predictions cluster tightly around the perfect prediction line, with slight deviations at higher sales values where data is sparser.

---

## Key Findings

- XGBoost with tuned hyperparameters is the best-performing model, achieving R² of 0.8692 — explaining nearly 87% of variance in global video game sales.
- Standard K-Fold cross-validation produced misleading results (Mean RMSE: 0.2470) due to imbalanced target distribution across folds. Stratified K-Fold resolved this, reducing Mean RMSE to 0.1379 with a standard deviation of 0.0024.
- ShuffleSplit independently confirmed generalization stability with Mean RMSE of 0.1416 and Std Dev of 0.0067.
- Engineered features — particularly `publisher_avg_sales` and `developer_avg_sales` — significantly improved model signal by encoding historical market performance.
- `critic_score` is a weak predictor of commercial success, suggesting that factors outside critical reception (marketing, franchise power, platform exclusivity) are dominant sales drivers.
- Total sales was intentionally excluded from the feature set to prevent data leakage, as it is a linear combination of the regional sales columns.

---

## Project Structure

    Video-Game-Sales-Forecasting/
    |-- ml-models-and-cross-validation.ipynb    # Full pipeline: preprocessing, EDA, modeling, cross-validation, deployment
    |-- sales_forecasting_xgb.pkl               # Serialized trained XGBoost model
    |-- README.md

---

## Technologies Used

- Python 3
- Pandas, NumPy — data manipulation, feature engineering, log transformation
- Matplotlib, Seaborn — sales distribution, correlation heatmap, feature importance, actual vs. predicted plots
- Scikit-learn — train-test split, Label Encoding, RandomForestRegressor, cross-validation (KFold, StratifiedKFold, ShuffleSplit), regression metrics
- XGBoost — gradient boosted regression, feature importance ranking
- LightGBM — lightweight gradient boosted regression for comparison
- Joblib — model serialization for deployment
- Jupyter Notebook — interactive development and documentation
- Kaggle — dataset source (VGChartz 2024) and execution environment

---

## Getting Started

1. Clone the repository:

       git clone https://github.com/sahandyousefi/Video-Game-Sales-Forecasting.git
       cd Video-Game-Sales-Forecasting

2. Install dependencies:

       pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib jupyter

3. Download the VGChartz 2024 dataset from Kaggle and place `vgchartz-2024.csv` in the input directory, or update the file path in the notebook to match your local setup.

4. Launch the notebook:

       jupyter notebook ml-models-and-cross-validation.ipynb

---

## Author

Sahand Yousefi
[GitHub](https://github.com/sahandyousefi) | [LinkedIn](https://www.linkedin.com/in/sahand-yousefi/)
