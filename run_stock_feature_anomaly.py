from __future__ import annotations

import argparse
from pathlib import Path

from src.stock_feature_anomaly import (
    StockFeatureAnomalyAnalyzer,
    load_single_stock_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find stock price anomalies and show which features dominate them."
    )
    parser.add_argument("--csv", required=True, help="Path to a single-stock CSV file.")
    parser.add_argument("--date-column", default="date", help="Name of the date column.")
    parser.add_argument("--price-column", default="close", help="Name of the stock price column.")
    parser.add_argument(
        "--volume-column",
        default="volume",
        help="Name of the volume column. Use 'none' if there is no volume column.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.03,
        help="Expected anomaly fraction. Try 0.01 to 0.05.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    volume_column = None if args.volume_column.lower() == "none" else args.volume_column

    data = load_single_stock_csv(args.csv, date_column=args.date_column)
    analyzer = StockFeatureAnomalyAnalyzer(contamination=args.contamination)
    result = analyzer.fit_predict(
        data=data,
        price_column=args.price_column,
        volume_column=volume_column,
    )

    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    result.scored_data.to_csv(output_dir / "stock_feature_scored_days.csv")
    result.top_anomalies.to_csv(output_dir / "stock_feature_top_anomalies.csv")

    print("Stock Feature Dominance Anomaly Analysis")
    print("=" * 55)
    print(f"Rows used: {len(result.scored_data):,}")
    print(f"Features analyzed: {len(result.feature_columns):,}")
    print()
    print("Top 10 price-feature anomaly days:")
    for date, row in result.top_anomalies.head(10).iterrows():
        price_return = row.get("price_return", 0.0)
        print(f"\n{date.date()} | score={row['anomaly_score']:.3f} | return={price_return:+.2%}")
        print(row["dominant_features"])

    print("\nSaved CSV files in examples/:")
    print("- stock_feature_scored_days.csv")
    print("- stock_feature_top_anomalies.csv")


if __name__ == "__main__":
    main()
