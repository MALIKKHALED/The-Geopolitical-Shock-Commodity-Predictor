# The "Geopolitical Shock" Commodity Predictor

Predicting Oil & Gold price movements from geopolitical news signals using NLP and Machine Learning.

---

## Overview

**The problem:**
Oil and Gold prices react instantly to geopolitical fear. When political leaders make aggressive speeches or countries announce sanctions, commodity prices spike. Traders need to react before the market fully digests the news.

**The approach:**
This project combines Natural Language Processing (NLP) with time-series financial analysis. We use historical news headlines to extract geopolitical risk signals, then build a machine learning model to predict whether commodity prices will experience abnormal returns in the days following high-risk news events.

**The goal:**
Build a "Commodity Risk Radar" — a web application where users paste a breaking news article and get an instant risk assessment with a historical-data-backed prediction.

---

## Key Features

- Fine-tuned BERT model for geopolitical risk classification (not just generic sentiment)
- Hybrid NLP approach: deep learning sentiment + keyword-based geopolitical risk index
- Statistical rigor: Granger causality tests, event study methodology, lag correlation analysis
- XGBoost prediction model with logistic regression baseline for comparison
- Target: Abnormal returns / volatility spikes (not naive up/down prediction)
- Interactive Streamlit web application for real-time risk assessment

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
│   ├── 04_nlp_pipeline.ipynb           # (coming soon)
│   ├── 05_feature_engineering.ipynb    # (coming soon)
│   ├── 06_modeling.ipynb               # (coming soon)
│   └── 07_evaluation.ipynb             # (coming soon)
│
├── src/                      # Reusable Python modules
├── app/                      # Streamlit web app
├── models/                   # Saved trained models
├── reports/                  # Generated figures and analysis
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

- [x] **Phase 1: Data Collection** — Download news headlines from Kaggle, pull Oil & Gold prices via yfinance, validate all datasets
- [x] **Phase 2: Data Cleaning & Merging** — Standardize dates, handle weekends/holidays, aggregate daily news, merge into a single master dataset
- [x] **Phase 3: Exploratory Data Analysis (EDA)** — Timeline overlays, keyword spike analysis, lag correlations, Granger causality tests, event studies
- [ ] **Phase 4: NLP Pipeline** — Fine-tune BERT for geopolitical risk classification, build keyword-based risk index, score each day
- [ ] **Phase 5: Feature Engineering** — Create model-ready features: sentiment scores, keyword frequencies, price lags, volatility metrics
- [ ] **Phase 6: Modeling & Evaluation** — Train XGBoost + logistic regression baseline, evaluate with proper metrics, analyze feature importance
- [ ] **Phase 7: Web Application** — Build "Commodity Risk Radar" in Streamlit for real-time risk assessment

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

5. **Run the data collection notebook:**
   - Open `notebooks/01_data_collection.ipynb` and run all cells
   - This will download Oil & Gold price data via yfinance and save them to `data/raw/`

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **NLP** | HuggingFace Transformers (BERT fine-tuning) |
| **Machine Learning** | XGBoost, scikit-learn |
| **Data Processing** | pandas, NumPy, yfinance |
| **Statistics** | scipy, statsmodels |
| **Visualization** | matplotlib, seaborn |
| **Web App** | Streamlit |
| **Deep Learning** | PyTorch |

---

## Current Status

> **Phase 3 — Exploratory Data Analysis: ✅ Complete**

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
Key statistical findings from the EDA:
- **Fat tails confirmed:** Oil has 215 days with >5% moves (high kurtosis) — enough extreme events to model
- **Volatility clusters** around known geopolitical events (2008 crisis, COVID 2020, Iraq War, Arab Spring)
- **News volume normalization:** Yearly z-score applied to handle declining headline count post-2016
- **Lag correlation:** Gold shows significant lag-3 effect (delayed flight-to-safety). Oil shows lag-4 mean reversion
- **Event study:** High-news days (z > 2) show same-day price impact but limited delayed effect with raw counts
- **Granger causality: ✅ Statistically proven** — news volume Granger-causes commodity price movements, especially for gold
- **Key insight:** Raw headline count is a weak but real signal. NLP-based risk scoring (Phase 4) is expected to significantly strengthen the predictive power

**Next up:** Phase 4 — NLP Pipeline (geopolitical risk scoring using keyword analysis and BERT).

---

## Malik Khalid

