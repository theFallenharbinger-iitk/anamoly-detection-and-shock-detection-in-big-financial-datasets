from __future__ import annotations

import argparse
from pathlib import Path

from src.detector import ExplainableAnomalyDetector
from src.real_data import DEFAULT_TICKERS, download_stock_returns, simple_feature_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anomaly detection on real stock market data from yfinance."
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated tickers, for example AAPL,MSFT,NVDA,SPY,QQQ.",
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="Optional end date in YYYY-MM-DD format.")
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.03,
        help="Expected anomaly fraction. Try 0.01 to 0.05.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Download timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]

    print("Downloading real market data...")
    returns = download_stock_returns(
        tickers=tickers,
        start=args.start,
        end=args.end,
        timeout=args.timeout,
    )

    detector = ExplainableAnomalyDetector(contamination=args.contamination)
    result = detector.fit_predict(returns, simple_feature_groups(list(returns.columns)))

    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    result.scored_data.to_csv(output_dir / "real_market_scored_days.csv")
    result.top_anomalies.to_csv(output_dir / "real_market_top_anomalies.csv")

    print("Real Market Anomaly Detection Demo")
    print("=" * 55)
    print(f"Downloaded {returns.shape[0]} trading days and {returns.shape[1]} tickers.")
    print(detector.pca_summary)
    print()
    print("Top 10 unusual market days:")
    for date, row in result.top_anomalies.head(10).iterrows():
        print(f"\n{date.date()} | score={row['anomaly_score']:.3f}")
        print(row["plain_explanation"])

    print("\nSaved CSV files in examples/:")
    print("- real_market_scored_days.csv")
    print("- real_market_top_anomalies.csv")


if __name__ == "__main__":
    main()
