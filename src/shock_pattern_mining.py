from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


@dataclass
class ShockPatternResult:
    periods: pd.DataFrame
    similarities: pd.DataFrame


def mine_shock_patterns(
    scored_data: pd.DataFrame,
    feature_columns: list[str],
    max_gap: int = 3,
    top_features: int = 5,
) -> ShockPatternResult:
    """Group flagged days into periods and describe repeated feature patterns."""

    if "model_flag" not in scored_data.columns:
        raise ValueError("scored_data must contain a model_flag column.")

    flagged_positions = np.flatnonzero(scored_data["model_flag"].astype(bool).to_numpy())
    if len(flagged_positions) == 0:
        return ShockPatternResult(periods=pd.DataFrame(), similarities=pd.DataFrame())

    groups = _group_flagged_positions(flagged_positions, max_gap=max_gap)
    contribution_columns = [f"contribution_{feature}" for feature in feature_columns]

    z_values = StandardScaler().fit_transform(scored_data[feature_columns])
    z_frame = pd.DataFrame(z_values, index=scored_data.index, columns=feature_columns)

    period_rows: list[dict[str, object]] = []
    signatures: list[np.ndarray] = []

    for period_id, positions in enumerate(groups, start=1):
        period = scored_data.iloc[positions]
        period_z = z_frame.iloc[positions]
        mean_contribution = period[contribution_columns].mean()
        mean_z = period_z.mean()

        top_contribution_columns = mean_contribution.sort_values(ascending=False).head(top_features)
        top_feature_names = [
            column.replace("contribution_", "", 1)
            for column in top_contribution_columns.index
        ]

        labels = _pattern_labels(period, top_feature_names, mean_z)
        feature_story = _feature_story(period, top_feature_names, mean_z)
        price_move = float((1 + period["price_return"]).prod() - 1)

        signature = np.concatenate(
            [
                mean_contribution.to_numpy(dtype=float),
                np.clip(mean_z[feature_columns].to_numpy(dtype=float) / 3.0, -1.0, 1.0),
            ]
        )
        signatures.append(signature)

        period_rows.append(
            {
                "period_id": period_id,
                "start_date": period.index.min(),
                "end_date": period.index.max(),
                "days_in_period": len(period),
                "price_move": price_move,
                "max_anomaly_score": float(period["anomaly_score"].max()),
                "average_anomaly_score": float(period["anomaly_score"].mean()),
                "dominant_features": ", ".join(top_feature_names),
                "pattern_labels": ", ".join(labels),
                "feature_story": feature_story,
            }
        )

    periods = pd.DataFrame(period_rows)
    similarities = _similar_periods(periods, signatures)
    return ShockPatternResult(periods=periods, similarities=similarities)


def _group_flagged_positions(flagged_positions: np.ndarray, max_gap: int) -> list[np.ndarray]:
    groups: list[list[int]] = [[int(flagged_positions[0])]]
    for position in flagged_positions[1:]:
        if int(position) - groups[-1][-1] <= max_gap:
            groups[-1].append(int(position))
        else:
            groups.append([int(position)])
    return [np.array(group, dtype=int) for group in groups]


def _pattern_labels(
    period: pd.DataFrame,
    top_feature_names: list[str],
    mean_z: pd.Series,
) -> list[str]:
    labels: list[str] = []
    price_move = float((1 + period["price_return"]).prod() - 1)
    average_abs_return = float(period["absolute_return"].mean())

    if price_move <= -0.08:
        labels.append("stock selloff / plummeting price")
    elif price_move >= 0.08:
        labels.append("stock spike or rebound")
    elif average_abs_return >= 0.04:
        labels.append("violent one-day price movement")

    if any("volatility" in feature for feature in top_feature_names):
        volatility_features = [feature for feature in top_feature_names if "volatility" in feature]
        if any(mean_z.get(feature, 0.0) > 1.0 for feature in volatility_features):
            labels.append("volatility burst")

    if any("volume" in feature for feature in top_feature_names):
        volume_features = [feature for feature in top_feature_names if "volume" in feature]
        if any(mean_z.get(feature, 0.0) > 1.0 for feature in volume_features):
            labels.append("volume shock")

    pressure_features = [
        feature
        for feature in top_feature_names
        if "market_return" in feature or "peer" in feature
    ]
    if pressure_features:
        pressure_mean = float(period[pressure_features].mean().mean())
        if pressure_mean < -0.01:
            labels.append("market or peer selloff pressure")
        elif pressure_mean > 0.01:
            labels.append("market or peer rally pressure")

    if "input_market_return" in period.columns:
        market_move = float(period["input_market_return"].mean())
        if price_move < -0.04 and market_move > 0:
            labels.append("stock fell while market was positive")
        elif price_move > 0.04 and market_move < 0:
            labels.append("stock rose while market was negative")

    if not labels:
        labels.append("unusual mixed feature pattern")
    return labels


def _feature_story(
    period: pd.DataFrame,
    top_feature_names: list[str],
    mean_z: pd.Series,
) -> str:
    parts: list[str] = []
    for feature in top_feature_names:
        z_value = float(mean_z.get(feature, 0.0))
        raw_value = float(period[feature].mean())

        if z_value >= 1.0:
            level = "above its normal range"
        elif z_value <= -1.0:
            level = "below its normal range"
        else:
            level = "near normal but still important relative to the other features"

        direction = _raw_direction(feature, raw_value)
        if direction:
            parts.append(f"{feature} was {direction} and {level}")
        else:
            parts.append(f"{feature} was {level}")

    return "; ".join(parts) + "."


def _raw_direction(feature: str, raw_value: float) -> str:
    if "return" in feature or "gap" in feature or "change" in feature:
        if raw_value <= -0.02:
            return "negative / falling"
        if raw_value >= 0.02:
            return "positive / rising"
    if "volatility" in feature and raw_value > 0:
        return "elevated"
    if "volume" in feature:
        if raw_value <= -0.20:
            return "unusually low"
        if raw_value >= 0.20:
            return "unusually high"
    return ""


def _similar_periods(periods: pd.DataFrame, signatures: list[np.ndarray]) -> pd.DataFrame:
    if len(signatures) < 2:
        return pd.DataFrame()

    matrix = np.vstack(signatures)
    similarity_matrix = cosine_similarity(matrix)
    rows: list[dict[str, object]] = []

    for i, period in periods.iterrows():
        candidates = [
            (j, float(similarity_matrix[i, j]))
            for j in range(len(periods))
            if j != i
        ]
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:3]
        for rank, (j, score) in enumerate(candidates, start=1):
            similar = periods.iloc[j]
            rows.append(
                {
                    "period_id": int(period["period_id"]),
                    "period_start": period["start_date"],
                    "period_end": period["end_date"],
                    "similar_rank": rank,
                    "similar_period_id": int(similar["period_id"]),
                    "similar_start": similar["start_date"],
                    "similar_end": similar["end_date"],
                    "similarity_score": score,
                    "shared_pattern": similar["pattern_labels"],
                }
            )

    return pd.DataFrame(rows)
