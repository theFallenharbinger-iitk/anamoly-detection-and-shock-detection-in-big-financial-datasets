from __future__ import annotations

import pandas as pd


def evaluate_against_planted_events(scored_data: pd.DataFrame) -> dict[str, float | int]:
    """Compare model flags with the planted answer key in synthetic data."""

    if "planted_anomaly" not in scored_data.columns:
        return {}

    predicted = scored_data["model_flag"].astype(bool)
    actual = scored_data["planted_anomaly"].astype(bool)

    true_positive = int((predicted & actual).sum())
    false_positive = int((predicted & ~actual).sum())
    false_negative = int((~predicted & actual).sum())

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    return {
        "planted_anomalies": int(actual.sum()),
        "model_flags": int(predicted.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }
