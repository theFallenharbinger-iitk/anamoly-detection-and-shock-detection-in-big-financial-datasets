from __future__ import annotations

import argparse
from pathlib import Path

from src.detector import ExplainableAnomalyDetector
from src.real_data import load_market_csv, simple_feature_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anomaly detection on a local CSV file."
    )
    parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    parser.add_argument("--date-column", default="date", help="Name of the date column.")
    parser.add_argument(
        "--input-kind",
        choices=["prices", "returns"],
        default="prices",
        help="Use 'prices' for price columns or 'returns' for return columns.",
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
    returns = load_market_csv(
        csv_path=args.csv,
        date_column=args.date_column,
        input_kind=args.input_kind,
    )

    detector = ExplainableAnomalyDetector(contamination=args.contamination)
    result = detector.fit_predict(returns, simple_feature_groups(list(returns.columns)))

    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    result.scored_data.to_csv(output_dir / "csv_scored_days.csv")
    result.top_anomalies.to_csv(output_dir / "csv_top_anomalies.csv")

    print("CSV Anomaly Detection Demo")
    print("=" * 55)
    print(f"Loaded {returns.shape[0]} rows and {returns.shape[1]} numeric features.")
    print(detector.pca_summary)
    print()
    print("Top 10 unusual rows:")
    for date, row in result.top_anomalies.head(10).iterrows():
        print(f"\n{date.date()} | score={row['anomaly_score']:.3f}")
        print(row["plain_explanation"])

    print("\nSaved CSV files in examples/:")
    print("- csv_scored_days.csv")
    print("- csv_top_anomalies.csv")


if __name__ == "__main__":
    main()
