# Project Explanation In Human Language

## One-Line Explanation

This project finds unusual days in high-dimensional financial data and explains which features made those days unusual.

## The Problem

In many real datasets, there are no labels.

That means we do not know in advance which rows are normal and which rows are abnormal. This is common in finance, cybersecurity, sensors, and fraud detection.

The challenge becomes:

> Can we find strange behavior without being told exactly what strange behavior looks like?

## Why High-Dimensional Data Is Hard

High-dimensional data means each row has many features.

For example, one day of financial data might include returns for 80 stocks. A human cannot easily inspect all 80 values at once.

Also, in high dimensions, normal distance-based methods become weaker because many points start looking far apart from each other.

## My Solution

I built a simple pipeline:

1. Generate realistic financial data.
2. Plant a few unusual events.
3. Compress the data using PCA.
4. Detect anomalies using Isolation Forest.
5. Explain each flagged day using feature-level deviations.

I also added an LSTM Autoencoder for sequence anomalies.

That model looks at windows of days instead of only one day at a time.

I also added a single-stock feature-dominance analyzer.

That part answers:

> When one stock has an anomalous price move, which features explain most of the anomaly?

Then I added historical shock pattern mining.

That part answers:

> Across all historical price shocks, do similar feature patterns repeat?

## What PCA Does

PCA reduces many features into fewer summary features.

A simple way to explain it:

> PCA removes repeated noise and keeps the main patterns in the data.

In this project, PCA helps the model focus on the important structure instead of getting distracted by many noisy columns.

## What Isolation Forest Does

Isolation Forest is based on a simple idea:

> Weird points are easier to isolate than normal points.

If a day behaves very differently from the rest, the model can separate it quickly. That day receives a higher anomaly score.

## How Explanations Work

After the model flags a day, the project checks which assets moved furthest from their usual behavior.

This turns a model output into something understandable:

> The model flagged this day because many assets moved unusually together.

or:

> The model flagged this day because one asset moved far more than normal.

For the LSTM Autoencoder, the explanation is slightly different:

> The model flagged this window because it could not reconstruct the sequence well. The largest reconstruction errors came from these assets.

That means the model is saying:

> This short time period does not look like the patterns I learned from the rest of the data.

For the single-stock feature analyzer, the explanation is:

> This stock was anomalous mostly because these features were unusually far from normal.

For example, if `absolute_return` contributes 34% and `market_return` contributes 14%, then the anomaly was mostly driven by the stock's own large move, with some contribution from the broader market.

The shock pattern miner groups nearby anomaly days into periods. Then it gives each period a label.

Example labels:

- plummeting price
- rebound or spike
- volatility burst
- volume shock
- peer-stock selloff pressure
- market divergence

This is useful because it moves the project from detection to pattern discovery.

Instead of only saying:

> This day was anomalous.

The project can now say:

> This historical shock looks similar to previous selloff periods where volatility rose and peer stocks were also falling.

## Why Synthetic Data Is Used First

Synthetic data is useful because I can plant known anomalies.

That gives me a hidden answer key. I can check whether the unsupervised model found the events I planted, even though the model never saw those labels during training.

## What I Would Improve Next

The next version would use real market data and compare multiple methods:

- Isolation Forest
- LSTM Autoencoder
- Autoencoder
- Local Outlier Factor
- One-Class SVM
- EVT-based thresholding
- SHAP-style feature explanations
- clustering shock periods into market regimes

The research version would study which method works best for different anomaly types.
