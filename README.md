# Explainable Stock Anomaly Detection And Shock Pattern Mining

This is a working Python prototype for finding unusual stock-market behavior, explaining which features caused the anomaly, and checking whether similar shock patterns appeared in history.

In simple words:

> The project detects strange market days, explains which features made them strange, groups nearby shocks into historical periods, and finds whether similar feature patterns repeated before.

## Table Of Contents

- [What This Project Does](#what-this-project-does)
- [Why This Matters](#why-this-matters)
- [Models Used](#models-used)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Formats](#data-formats)
- [Real Data Sources](#real-data-sources)
- [Verified Results](#verified-results)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [How To Explain This In An Interview](#how-to-explain-this-in-an-interview)
- [Resume Bullet](#resume-bullet)
- [Research Significance](#research-significance)
- [Research Roadmap](#research-roadmap)
- [Current Limitations](#current-limitations)

## What This Project Does

The project has four connected layers.

1. **High-dimensional anomaly detection**

   Finds unusual market days across many assets using PCA and Isolation Forest.

2. **Sequence anomaly detection**

   Uses an LSTM Autoencoder to find unusual windows of days, not just isolated one-day outliers.

3. **Single-stock feature dominance**

   Given one stock's price and features, identifies which features dominate the price anomaly.

4. **Historical shock pattern mining**

   Groups nearby anomaly days into shock periods, labels their behavior, and finds similar past shock periods.

## Why This Matters

Most real anomaly detection problems do not come with labels.

In finance, nobody gives you a perfect answer key that says:

- this day was a crash
- this day was a volume shock
- this day was a peer-driven selloff
- this day was just noise

So the model has to learn what normal behavior looks like and then flag rare behavior. The important part is not only detecting the anomaly, but also explaining it in human language.

This project is useful because it moves from:

> Something weird happened.

to:

> Something weird happened, these features dominated it, and this shock looks similar to these previous historical shocks.

## Models Used

### 1. PCA + Isolation Forest

This is the baseline model.

Pipeline:

1. Standardize all features.
2. Use PCA to reduce noisy high-dimensional data.
3. Use Isolation Forest to score unusual rows.
4. Explain each anomaly using feature-level deviations.

Human explanation:

> PCA removes repeated noise and keeps the main patterns. Isolation Forest then finds rows that are easier to separate from normal rows.

Best for:

- fast anomaly detection
- point anomalies
- interpretable baseline results
- resume-friendly ML explanation

### 2. LSTM Autoencoder

This is the sequence model.

Pipeline:

1. Convert data into rolling windows of days.
2. Train an LSTM Autoencoder to reconstruct those windows.
3. Use reconstruction error as the anomaly score.
4. Explain which features contributed most to reconstruction error.

Human explanation:

> The LSTM Autoencoder learns what normal sequences look like. If it cannot rebuild a window well, that window is probably unusual.

Best for:

- market stress windows
- volatility regimes
- multi-day shocks
- research expansion

### 3. Single-Stock Feature Dominance

This answers:

> Given one stock price and several features, which features mostly explain the price anomaly?

Example features:

- price return
- absolute price return
- rolling volatility
- moving-average gap
- volume change
- volume compared with 20-day average
- market return
- peer stock return
- sentiment score
- macro or fundamental features

Human explanation:

> A stock price anomaly is explained by ranking which standardized features were furthest from normal during that anomaly.

### 4. Historical Shock Pattern Mining

This answers:

> Across all historical price shocks, did similar feature patterns repeat?

The system groups nearby anomaly days into shock periods and labels them.

Example labels:

- stock selloff / plummeting price
- stock spike or rebound
- volatility burst
- volume shock
- market or peer selloff pressure
- unusual mixed feature pattern

Human explanation:

> Instead of treating every anomaly as separate, the system checks whether shocks form recurring historical patterns.

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the baseline synthetic demo:

```bash
python run_demo.py
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

Run the LSTM Autoencoder:

```bash
python run_lstm_autoencoder.py --source synthetic --epochs 20
```

Run single-stock feature dominance:

```bash
python run_stock_feature_anomaly.py --csv examples/aapl_feature_sample.csv --date-column date --price-column close --volume-column volume
```

Run historical shock pattern mining:

```bash
python run_shock_pattern_mining.py --csv examples/aapl_feature_sample.csv --date-column date --price-column close --volume-column volume
```

## Data Formats

### Multi-Stock Price CSV

Use this for general anomaly detection across many assets.

```text
date,AAPL,MSFT,NVDA,SPY,QQQ
2020-01-02,72.8,154.2,59.9,300.2,214.3
2020-01-03,72.1,152.8,58.8,297.4,211.9
```

Run:

```bash
python run_csv_data.py --csv path/to/your_file.csv --date-column date --input-kind prices
```

If the file already contains returns:

```bash
python run_csv_data.py --csv path/to/your_file.csv --date-column date --input-kind returns
```

### Single-Stock Feature CSV

Use this when you want to know which features explain a specific stock's price anomaly.

```text
date,close,volume,market_return,peer_msft_return,peer_nvda_return
2020-01-02,72.8,135480400,0.004,0.018,0.012
2020-01-03,72.1,146322800,-0.006,-0.012,-0.009
```

Run:

```bash
python run_stock_feature_anomaly.py --csv path/to/stock_features.csv --date-column date --price-column close --volume-column volume
```

Then mine repeated historical shock patterns:

```bash
python run_shock_pattern_mining.py --csv path/to/stock_features.csv --date-column date --price-column close --volume-column volume
```

## Real Data Sources

The project works without external data because it can generate synthetic data. For real testing, use:

- **Yahoo Finance / yfinance** for stock prices
- **Kaggle** for ready-made finance and anomaly datasets
- **Nasdaq Data Link** for professional finance and economic datasets
- **FRED** for macroeconomic indicators
- **UCI Machine Learning Repository** for general anomaly-detection datasets

Fast yfinance test:

```bash
python run_real_data.py --tickers AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,XOM,SPY,QQQ --start 2020-01-01
```

If the API is slow or blocked, download a CSV manually and use `run_csv_data.py`.

## Verified Results

### Synthetic Baseline

The synthetic demo plants known anomalies such as market crashes, sector shocks, volatility bursts, and single-asset spikes.

Verified result:

```text
precision: 0.722
recall: 0.929
```

### Real Market Test

Tested on real adjusted close data for:

```text
AAPL, MSFT, NVDA, AMZN
```

Date range:

```text
2020-01-01 to 2026-06-26
```

The baseline model flagged major unusual days including March 2020 market-stress dates.

Example:

```text
2020-03-16
MSFT -7.9x usual
AAPL -6.6x usual
NVDA -5.7x usual
```

### LSTM Autoencoder Real-Data Result

The LSTM Autoencoder flagged market-stress windows ending around late March and early April 2020.

Example:

```text
Window ending 2020-03-24
MSFT: 39% of reconstruction error
AAPL: 32% of reconstruction error
NVDA: 19% of reconstruction error
AMZN: 10% of reconstruction error
```

Human explanation:

> The window was unusual because the model could not reconstruct the sequence well, mainly due to MSFT, AAPL, and NVDA behavior.

### Single-Stock Feature Dominance Result

Tested on an AAPL feature sample.

Example:

```text
2020-03-16 | return=-12.86%
Dominant features:
absolute_return 21%
peer_msft_return 20%
rolling_10d_volatility 12%
market_return 12%
peer_nvda_return 11%
```

Human explanation:

> The stock anomaly was mostly driven by the size of AAPL's own move, peer-stock movement, market movement, and elevated volatility.

### Historical Shock Pattern Result

The shock pattern miner found:

```text
13 historical shock periods
```

Example:

```text
2020-09-03 to 2020-09-09
price move: -10.77%
pattern: stock selloff / plummeting price, market or peer selloff pressure
top features: absolute_return, peer_msft_return, rolling_5d_return, market_return, moving_average_gap_10d
```

Human explanation:

> This was a short selloff where the stock fell, peer stocks were weak, the market feature was negative, and the stock moved below its recent trend.

## Output Files

Generated outputs are saved in `examples/` and ignored by Git.

Common outputs:

- `scored_days.csv`
- `top_anomalies.csv`
- `real_market_top_anomalies.csv`
- `csv_top_anomalies.csv`
- `lstm_top_anomalies.csv`
- `stock_feature_top_anomalies.csv`
- `shock_pattern_periods.csv`
- `shock_pattern_similarities.csv`

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── examples/
│   └── README.md
├── src/
│   ├── detector.py
│   ├── evaluation.py
│   ├── lstm_autoencoder.py
│   ├── real_data.py
│   ├── shock_pattern_mining.py
│   ├── stock_feature_anomaly.py
│   └── synthetic_data.py
├── DATA_SOURCES.md
├── PROJECT_EXPLANATION.md
├── RESUME_NOTES.md
├── run_csv_data.py
├── run_demo.py
├── run_lstm_autoencoder.py
├── run_real_data.py
├── run_shock_pattern_mining.py
├── run_stock_feature_anomaly.py
├── requirements.txt
└── README.md
```

## How To Explain This In An Interview

Short version:

> I built an explainable anomaly detection system for financial data. It detects unusual market behavior, explains which features caused the anomaly, and groups historical price shocks into recurring patterns.

Longer version:

> I started with a PCA plus Isolation Forest baseline to detect unusual high-dimensional market days. Then I added an LSTM Autoencoder to detect sequence anomalies across rolling windows. After that, I built a single-stock feature-dominance analyzer to show which features explain a stock price anomaly. Finally, I added shock pattern mining, which groups nearby anomaly days into historical periods and checks whether similar feature patterns appeared before.

## Resume Bullet

Built an explainable unsupervised anomaly detection system for financial data using PCA, Isolation Forest, an LSTM Autoencoder, real-market data ingestion, single-stock feature-dominance analysis, and historical shock-pattern mining to identify recurring drivers of stock price anomalies.

## Research Significance

The research significance is:

> Detecting anomalies is useful, but explaining and comparing them historically is more valuable.

This project studies whether stock price shocks have recurring feature signatures, such as:

- price plummeting with peer-stock weakness
- high volatility with volume shock
- stock-specific spike while market features stay normal
- broad market stress across multiple assets
- repeated selloff patterns across different historical periods

That makes the project expandable into a 2-3 month research project.

## Research Roadmap

Possible next steps:

- add SHAP-style explanations
- cluster shock periods into market regimes
- compare Isolation Forest, One-Class SVM, LOF, Autoencoder, and LSTM Autoencoder
- add macroeconomic features from FRED
- add sentiment or news features
- use EVT or conformal prediction for better thresholds
- write a research-style report comparing anomaly types

## Current Limitations

This is a prototype, not a trading system.

Current limitations:

- real financial anomalies do not have perfect labels
- feature dominance is based on standardized deviation, not causal proof
- historical similarity means pattern similarity, not guaranteed future behavior
- the LSTM Autoencoder is intentionally small so it can run on a laptop

The goal is to create a clean, explainable foundation that can later become a deeper research project.
