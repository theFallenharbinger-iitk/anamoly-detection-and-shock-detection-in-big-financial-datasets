# Resume Notes

## Project Title Options

- Explainable Unsupervised Anomaly Detection
- High-Dimensional Financial Anomaly Detection
- Explainable Market Anomaly Detection Dashboard

## Best Resume Bullet

Built an explainable unsupervised anomaly detection prototype for high-dimensional financial data using PCA, Isolation Forest, an LSTM Autoencoder, synthetic anomaly injection, real-market data testing, CSV-based data ingestion, single-stock feature-dominance analysis, historical shock-pattern mining, and a Streamlit dashboard for interactive inspection.

## Stronger Technical Version

Developed a Python anomaly detection pipeline for high-dimensional financial return data using standardization, PCA-based dimensionality reduction, Isolation Forest scoring, LSTM Autoencoder reconstruction error, synthetic anomaly injection, yfinance data ingestion, local CSV support, interpretable feature-contribution summaries, and historical pattern mining across stock price shock periods.

## Interview Pitch

I built this because many anomaly detection problems do not have labels. The model has to learn what normal behavior looks like and then flag rare patterns. I used synthetic financial data first because it lets me plant known anomalies and evaluate the system. Then I added real-data testing through yfinance and CSV files. The baseline uses PCA to reduce noise and Isolation Forest to find unusual single days. The upgraded model uses an LSTM Autoencoder to detect unusual windows of days using reconstruction error. I also added a single-stock analyzer that shows which features dominate a price anomaly, such as return size, volume shock, volatility, market return, or peer-stock movement. Finally, I added pattern mining to group historical shock periods and find whether similar shocks had similar feature behavior.
