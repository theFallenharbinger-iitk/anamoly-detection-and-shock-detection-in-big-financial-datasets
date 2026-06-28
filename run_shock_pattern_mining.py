from __future__ import annotations

import argparse
from pathlib import Path

from src.shock_pattern_mining import mine_shock_patterns
from src.stock_feature_anomaly import StockFeatureAnomalyAnalyzer, load_single_stock_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find repeated feature patterns across historical stock price shocks."
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
    parser.add_argument(
        "--max-gap",
        type=int,
        default=3,
        help="Join flagged days into the same period if they are this many rows apart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    volume_column = None if args.volume_column.lower() == "none" else args.volume_column

    data = load_single_stock_csv(args.csv, date_column=args.date_column)
    analyzer = StockFeatureAnomalyAnalyzer(contamination=args.contamination)
    anomaly_result = analyzer.fit_predict(
        data=data,
        price_column=args.price_column,
        volume_column=volume_column,
    )
    pattern_result = mine_shock_patterns(
        scored_data=anomaly_result.scored_data,
        feature_columns=anomaly_result.feature_columns,
        max_gap=args.max_gap,
    )

    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    anomaly_result.scored_data.to_csv(output_dir / "shock_pattern_scored_days.csv")
    pattern_result.periods.to_csv(output_dir / "shock_pattern_periods.csv", index=False)
    pattern_result.similarities.to_csv(output_dir / "shock_pattern_similarities.csv", index=False)

    print("Historical Price Shock Pattern Mining")
    print("=" * 55)
    print(f"Rows used: {len(anomaly_result.scored_data):,}")
    print(f"Shock periods found: {len(pattern_result.periods):,}")
    print()

    for _, row in pattern_result.periods.head(10).iterrows():
        print(
            f"Period {int(row['period_id'])}: "
            f"{row['start_date'].date()} to {row['end_date'].date()} "
            f"| price move={row['price_move']:+.2%}"
        )
        print(f"Pattern: {row['pattern_labels']}")
        print(f"Top features: {row['dominant_features']}")
        print(f"Feature behavior: {row['feature_story']}")

        similar = pattern_result.similarities[
            pattern_result.similarities["period_id"] == row["period_id"]
        ].head(2)
        if not similar.empty:
            similar_text = [
                f"{item.similar_start.date()} to {item.similar_end.date()} ({item.similarity_score:.2f})"
                for item in similar.itertuples()
            ]
            print(f"Most similar historical periods: {', '.join(similar_text)}")
        print()

    print("Saved CSV files in examples/:")
    print("- shock_pattern_scored_days.csv")
    print("- shock_pattern_periods.csv")
    print("- shock_pattern_similarities.csv")


if __name__ == "__main__":
    main()
