# The "Geopolitical Shock" Commodity Predictor

Predicting Oil & Gold price volatility spikes from geopolitical news signals using NLP and Machine Learning.

---

## Overview

**The problem:**
Oil and Gold prices react instantly to geopolitical fear. When political leaders make aggressive speeches or countries announce sanctions, commodity prices spike. Traders need to react before the market fully digests the news.

**The approach:**
This project combines Natural Language Processing (NLP) with time-series financial analysis. We use historical news headlines to extract geopolitical risk signals, then build machine learning models to predict whether commodity prices will experience abnormal volatility in the days following high-risk news events.

**The goal:**
Build and evaluate predictive models that can identify upcoming commodity price volatility spikes using news-derived NLP features, proving that geopolitical news signals contain actionable information for commodity markets.

---

## Key Features

- FinBERT financial sentiment model for geopolitical risk classification (not just generic sentiment)
- Hybrid NLP approach: deep learning sentiment + keyword-based geopolitical risk index
- Statistical rigor: Granger causality tests, event study methodology, lag correlation analysis
- XGBoost prediction model with logistic regression baseline for comparison
- Target: Volatility spikes (not naive up/down prediction)
- Comprehensive model evaluation: threshold tuning, ROC/PR curves, feature importance analysis

---

## Data Sources

| Dataset | Source | Description |
|---|---|---|
| News Headlines | [A Million News Headlines](https://www.kaggle.com/datasets/therohk/million-headlines) (Kaggle) | 1.2M+ ABC News headlines from 2003–2021 |
| Crude Oil Prices | yfinance API (ticker: `CL=F`) | Daily OHLCV futures data, 2003–2021 |
| Gold Prices | yfinance API (ticker: `GC=F`) | Daily OHLCV futures data, 2003–2021 |

---

## Project Structure

```
The-Geopolitical-Shock-Commodity-Predictor/
│
├── data/
│   ├── raw/                  # Original downloaded data (never modified)
│   ├── processed/            # Cleaned, merged, feature-engineered data
│   └── external/             # Supplementary data
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_nlp_pipeline.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_modeling.ipynb
│
├── models/                   # Saved trained models (.joblib)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

- [x] **Phase 1: Data Collection** — Download news headlines from Kaggle, pull Oil & Gold prices via yfinance, validate all datasets
- [x] **Phase 2: Data Cleaning & Merging** — Standardize dates, handle weekends/holidays, aggregate daily news, merge into a single master dataset
- [x] **Phase 3: Exploratory Data Analysis (EDA)** — Timeline overlays, keyword spike analysis, lag correlations, Granger causality tests, event studies
- [x] **Phase 4: NLP Pipeline** — Keyword-based geopolitical risk scoring + FinBERT financial sentiment on 1.2M headlines, combined into hybrid risk score
- [x] **Phase 5: Feature Engineering** — 44 engineered features: lag/rolling NLP scores, volatility metrics, price lags, day-of-week effects. Time-series train/test split
- [x] **Phase 6: Modeling & Evaluation** — Train XGBoost + logistic regression, threshold tuning, time-series cross-validation, feature importance analysis, comprehensive model comparison

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MALIKKHALED/The-Geopolitical-Shock-Commodity-Predictor.git
   cd The-Geopolitical-Shock-Commodity-Predictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the news dataset:**
   - Go to [A Million News Headlines](https://www.kaggle.com/datasets/therohk/million-headlines) on Kaggle
   - Download and extract `abcnews-date-text.csv` into `data/raw/`

5. **Run the notebooks in order:**
   - Start with `01_data_collection.ipynb` and proceed sequentially through `06_modeling.ipynb`

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **NLP** | HuggingFace Transformers (FinBERT) |
| **Machine Learning** | XGBoost, scikit-learn |
| **Data Processing** | pandas, NumPy, yfinance |
| **Statistics** | scipy, statsmodels |
| **Visualization** | matplotlib, seaborn |
| **Deep Learning** | PyTorch |

---

## Results

> **Project Complete — All 6 Phases Finished ✅**

### Phase 1 — Data Collection ✅
All three datasets successfully downloaded and validated:
- **1,244,184** news headlines spanning Feb 2003 – Dec 2021
- **4,775** trading days of Crude Oil price data
- **4,771** trading days of Gold price data

### Phase 2 — Data Cleaning & Merging ✅
All datasets cleaned, standardized, and merged into a single `master_dataset.csv`:
- Flattened yfinance multi-level column headers and prefixed with `oil_` / `gold_`
- Aggregated **1.2M headlines** into daily counts + concatenated text per day
- Rolled forward weekend/holiday news to the next trading day using `np.searchsorted`
- Final dataset: **4,770 trading days × 13 columns**, zero null values

### Phase 3 — Exploratory Data Analysis ✅
Key statistical findings:
- **Fat tails confirmed:** Oil has 215 days with >5% moves — enough extreme events to model
- **Volatility clusters** around known geopolitical events (2008, COVID, Iraq War)
- **Granger causality: ✅ Statistically proven** — news volume Granger-causes commodity price movements
- **Lag correlation:** Gold shows significant lag-3 effect, Oil shows lag-4 mean reversion

### Phase 4 — NLP Pipeline ✅
Two-layer NLP risk scoring applied to all 1.2M headlines:
- **Keyword scoring:** Dictionary of ~95 geopolitical terms across 6 categories. 12% of headlines contain risk keywords
- **FinBERT sentiment:** Processed all 1.2M headlines on GPU using `ProsusAI/finbert`. Mean fear score: 0.30, max: 0.97
- **Hybrid risk score:** Combined keyword + FinBERT signals with min-max normalization
- 5 new features added to master dataset

### Phase 5 — Feature Engineering ✅
Created model-ready feature matrix with **44 features** and time-series train/test split:
- **Target variable:** Binary volatility spike prediction (90th percentile absolute return, one day ahead). Oil threshold: 3.72%, Gold: 1.77%
- **Lag features (23):** 1-5 day lags for NLP risk scores, 1-3 day lags for returns/volumes
- **Rolling features (11):** 5-day and 20-day windows for risk, returns, and volatility. Risk change indicators
- **Time-series split:** Train on 3,800 days (2003–2018), test on 950 days (2018–2021). No future leakage
- **Correlation analysis:** Volatility clustering dominates (20-day vol: r=0.34 for oil). NLP features `finbert_neg_roll5` and `hybrid_risk_roll20` rank 9th-10th for oil predictions — a real but secondary signal

### Phase 6 — Modeling & Evaluation ✅
Trained and evaluated 4 model variants with comprehensive analysis:

**Models Trained:**
1. Logistic Regression (Naive) — simple baseline
2. Logistic Regression (Balanced) — class-weighted loss
3. XGBoost (Default) — initial hyperparameters
4. XGBoost (Tuned) — time-series cross-validated hyperparameter optimization

**Oil Volatility Prediction (Best: XGBoost Tuned):**

| Metric | Value |
|---|---|
| F1 Score | **0.4601** |
| AUC-ROC | **0.7724** |
| Precision | 46.2% |
| Recall | 45.8% |
| Optimal Threshold | 0.65 |

**Gold Volatility Prediction (Best: XGBoost Tuned):**

| Metric | Value |
|---|---|
| F1 Score | **0.2169** |
| AUC-ROC | **0.6726** |
| Precision | 17.1% |
| Recall | 29.5% |
| Optimal Threshold | 0.49 |

**Key Findings:**
- **Volatility clustering is the dominant predictor** — 20-day rolling volatility is #1 for both commodities
- **NLP features provide genuine secondary signal** — 8-9 of the top 15 features are NLP-derived for both oil and gold
- **Simpler trees generalize better** — XGBoost selected max_depth=2 (shallowest), confirming regularization is critical with limited positive examples
- **Threshold tuning is essential** — optimal thresholds ranged from 0.16 to 0.93 across models
- **Oil prediction is significantly stronger than Gold** — gold spikes are more market-driven than news-driven

**Saved Models:**
- `models/xgb_tuned_oil.joblib` — Best oil volatility prediction model
- `models/xgb_tuned_gold.joblib` — Best gold volatility prediction model

---

## Malik Khalid
