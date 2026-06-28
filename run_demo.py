from __future__ import annotations

from pathlib import Path

from src.detector import ExplainableAnomalyDetector
from src.evaluation import evaluate_against_planted_events
from src.synthetic_data import feature_groups, make_synthetic_financial_data


def main() -> None:
    dataset = make_synthetic_financial_data()
    feature_names = [
        column
        for column in dataset.data.columns
        if column not in {"planted_anomaly", "planted_type"}
    ]

    detector = ExplainableAnomalyDetector(contamination=0.02)
    result = detector.fit_predict(dataset.data, feature_groups(feature_names))
    metrics = evaluate_against_planted_events(result.scored_data)

    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    result.scored_data.to_csv(output_dir / "scored_days.csv")
    result.top_anomalies.to_csv(output_dir / "top_anomalies.csv")
    dataset.event_log.to_csv(output_dir / "planted_events.csv", index=False)

    print("Explainable Unsupervised Anomaly Detection Demo")
    print("=" * 55)
    print(detector.pca_summary)
    print()
    print("Synthetic test results:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    print()
    print("Top 5 model findings:")
    for date, row in result.top_anomalies.head(5).iterrows():
        print(f"\n{date.date()} | score={row['anomaly_score']:.3f} | planted={row['planted_type']}")
        print(row["plain_explanation"])

    print("\nSaved CSV files in examples/.")


if __name__ == "__main__":
    main()
