# Data Sources For The Next Version

The current project does not need external data. It generates synthetic financial data so anyone can clone the repo and run it immediately.

When you expand the project, use real data in this order.

## Best First Real Dataset: Yahoo Finance Through `yfinance`

Use this first because it is simple and works well for a student project.

Good use case:

- daily stock prices
- many tickers at once
- converting prices into returns
- testing whether the detector finds unusual market days

Example tickers to start with:

```text
AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, XOM, UNH, PG
```

Plain-language explanation:

> I used public price history, converted prices into daily returns, and treated each stock as one feature.

Link:

- [yfinance documentation](https://ranaroussi.github.io/yfinance/)

Important note:

`yfinance` is useful for research and educational work, but you should check Yahoo Finance terms before using the data commercially.

## Best Dataset Marketplace: Kaggle

Use Kaggle when you want a ready-made CSV.

Good use case:

- anomaly detection datasets
- stock market examples
- fraud-style examples
- quick experiments without writing data-download code

Links:

- [Kaggle stock market anomaly detection dataset](https://www.kaggle.com/datasets/korpionn/stock-market-anomaly-detection-dataset)
- [Kaggle financial anomaly data](https://www.kaggle.com/datasets/devondev/financial-anomaly-data)

Plain-language explanation:

> I used public benchmark-style data to compare how well my detector works beyond synthetic examples.

## Best Professional Finance Source: Nasdaq Data Link

Use this later, not first.

Good use case:

- more serious finance datasets
- economic datasets
- alternative data
- research expansion

Links:

- [Nasdaq Data Link getting started](https://docs.data.nasdaq.com/docs/getting-started)
- [Nasdaq Data Link Python guide](https://docs.data.nasdaq.com/docs/python)

Plain-language explanation:

> I used a structured financial data API so the project can move from a demo into a more research-like workflow.

## Best Macro Data Source: FRED

Use FRED if you want to detect unusual economic periods instead of only unusual stock days.

Good use case:

- interest rates
- inflation
- unemployment
- recession indicators
- macro regime changes

Links:

- [FRED homepage](https://fred.stlouisfed.org/)
- [FRED API documentation](https://fred.stlouisfed.org/docs/api/fred/)

Plain-language explanation:

> I added macroeconomic indicators so the detector can find abnormal economic regimes, not just abnormal stock returns.

## Best General ML Source: UCI Machine Learning Repository

Use UCI if you want to test the method outside finance.

Good use case:

- general anomaly detection experiments
- sensor-style datasets
- comparing finance data with non-finance data

Links:

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Smartphone anomaly detection dataset](https://archive.ics.uci.edu/dataset/613/smartphone%2Bdataset%2Bfor%2Banomaly%2Bdetection%2Bin%2Bcrowds)

Plain-language explanation:

> I tested whether the same anomaly detection idea works outside finance.

## My Recommendation

For your GitHub resume project:

1. Keep the synthetic dataset as the default.
2. Add `yfinance` as the first real-data option.
3. Use 20-50 stocks.
4. Convert adjusted close prices into daily percentage returns.
5. Run the same detector on the real returns.

That gives you a clean story:

> I built the prototype on synthetic data so I could measure whether it works, then tested it on real stock data to make the project more realistic.

## How To Test Real Data In This Project

### Fastest API Test

```bash
python run_real_data.py --tickers AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,XOM,SPY,QQQ --start 2020-01-01
```

If the API times out, use the CSV route below.

### Most Reliable CSV Test

Download a CSV from Yahoo Finance, Kaggle, Nasdaq Data Link, or another source.

Make the file look like this:

```text
date,AAPL,MSFT,NVDA,SPY,QQQ
2020-01-02,72.8,154.2,59.9,300.2,214.3
2020-01-03,72.1,152.8,58.8,297.4,211.9
```

Then run:

```bash
python run_csv_data.py --csv path/to/your_file.csv --date-column date --input-kind prices
```

If the numbers are already daily returns, run:

```bash
python run_csv_data.py --csv path/to/your_file.csv --date-column date --input-kind returns
```
