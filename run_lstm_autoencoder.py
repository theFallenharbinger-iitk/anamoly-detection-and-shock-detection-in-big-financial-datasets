from __future__ import annotations

import argparse
from pathlib import Path

from src.lstm_autoencoder import LSTMAutoencoderDetector
from src.real_data import load_market_csv
from src.synthetic_data import make_synthetic_financial_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an LSTM Autoencoder for sequence anomaly detection."
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "csv"],
        default="synthetic",
        help="Use generated synthetic data or a local CSV file.",
    )
    parser.add_argument("--csv", default=None, help="Path to CSV when --source csv is used.")
    parser.add_argument("--date-column", default="date", help="Name of the CSV date column.")
    parser.add_argument(
        "--input-kind",
        choices=["prices", "returns"],
        default="prices",
        help="Use 'prices' for price columns or 'returns' for return columns.",
    )
    parser.add_argument("--days", type=int, default=900, help="Synthetic data days.")
    parser.add_argument("--assets", type=int, default=40, help="Synthetic data assets.")
    parser.add_argument("--window-size", type=int, default=20, help="Days per LSTM window.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="LSTM hidden size.")
    parser.add_argument("--latent-dim", type=int, default=12, help="Compressed latent size.")
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.03,
        help="Expected fraction of anomalous windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "synthetic":
        dataset = make_synthetic_financial_data(days=args.days, assets=args.assets)
        data = dataset.data
        source_note = "synthetic data with planted sequence anomalies"
    else:
        if not args.csv:
            raise ValueError("--csv is required when --source csv is used.")
        data = load_market_csv(
            csv_path=args.csv,
            date_column=args.date_column,
            input_kind=args.input_kind,
        )
        source_note = f"CSV data from {args.csv}"

    detector = LSTMAutoencoderDetector(
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        contamination=args.contamination,
    )
    result = detector.fit_predict(data)

    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    result.scored_windows.to_csv(output_dir / "lstm_scored_windows.csv")
    result.top_anomalies.to_csv(output_dir / "lstm_top_anomalies.csv")

    print("LSTM Autoencoder Sequence Anomaly Detection")
    print("=" * 55)
    print(f"Source: {source_note}")
    print(f"Rows: {len(data):,}")
    print(f"Features: {len(detector.feature_names):,}")
    print(f"Window size: {args.window_size} days")
    print(f"Final training loss: {result.training_losses[-1]:.6f}")
    print()
    print("Top 10 unusual windows:")
    for date, row in result.top_anomalies.head(10).iterrows():
        label = ""
        if "planted_type_in_window" in row:
            label = f" | planted={row['planted_type_in_window']}"
        print(f"\nWindow ending {date.date()} | error={row['lstm_reconstruction_error']:.6f}{label}")
        print(row["plain_explanation"])

    print("\nSaved CSV files in examples/:")
    print("- lstm_scored_windows.csv")
    print("- lstm_top_anomalies.csv")


if __name__ == "__main__":
    main()
