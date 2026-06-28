from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class DetectionResult:
    scored_data: pd.DataFrame
    top_anomalies: pd.DataFrame


class ExplainableAnomalyDetector:
    """A simple unsupervised detector with readable explanations.

    The model does three things:
    1. Standardizes every feature so large-scale features do not dominate.
    2. Compresses the data with PCA to reduce high-dimensional noise.
    3. Uses Isolation Forest to find rows that are easier to separate from the rest.
    """

    def __init__(
        self,
        contamination: float = 0.02,
        pca_variance: float = 0.90,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.pca_variance = pca_variance
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance, svd_solver="full")
        self.model = IsolationForest(
            n_estimators=250,
            contamination=contamination,
            max_features=0.80,
            random_state=random_state,
        )
        self.feature_names: list[str] = []
        self.feature_groups: dict[str, str] = {}

    def fit_predict(
        self,
        data: pd.DataFrame,
        feature_groups: dict[str, str] | None = None,
    ) -> DetectionResult:
        self.feature_names = [
            column
            for column in data.columns
            if column not in {"planted_anomaly", "planted_type"}
        ]
        self.feature_groups = feature_groups or {name: "Unknown group" for name in self.feature_names}

        x = data[self.feature_names]
        x_scaled = self.scaler.fit_transform(x)
        x_pca = self.pca.fit_transform(x_scaled)

        raw_score = -self.model.fit(x_pca).score_samples(x_pca)
        severity = pd.Series(raw_score).rank(pct=True).to_numpy()
        model_flag = self.model.predict(x_pca) == -1

        scored = data.copy()
        scored["anomaly_score"] = raw_score
        scored["severity_percentile"] = severity
        scored["model_flag"] = model_flag
        scored["plain_explanation"] = [
            self.explain_row(row_index=i, scaled_row=x_scaled[i], flagged=model_flag[i])
            for i in range(len(scored))
        ]

        top = scored.sort_values("anomaly_score", ascending=False).head(15)
        return DetectionResult(scored_data=scored, top_anomalies=top)

    def explain_row(self, row_index: int, scaled_row: np.ndarray, flagged: bool) -> str:
        abs_z = np.abs(scaled_row)
        top_indices = np.argsort(abs_z)[-5:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]

        moved_features = [
            f"{self.feature_names[i]} ({scaled_row[i]:+.1f}x usual)"
            for i in top_indices
        ]

        unusual_count = int((abs_z > 2.5).sum())
        broad_share = unusual_count / max(len(abs_z), 1)

        groups = [self.feature_groups.get(feature, "Unknown group") for feature in top_features]
        most_common_group = max(set(groups), key=groups.count)

        if broad_share > 0.35:
            pattern = "This looks like a broad market event because many features moved unusually together."
        elif groups.count(most_common_group) >= 3:
            pattern = f"This looks like a group-level event because several top signals came from {most_common_group}."
        else:
            pattern = "This looks like a narrow event because only a few features drove most of the weirdness."

        prefix = "Flagged as unusual." if flagged else "Not flagged, but here is what stood out most."
        return f"{prefix} {pattern} Biggest contributors: {', '.join(moved_features)}."

    @property
    def pca_summary(self) -> str:
        components = getattr(self.pca, "n_components_", None)
        if components is None:
            return "PCA has not been fitted yet."
        variance = self.pca.explained_variance_ratio_.sum()
        return (
            f"PCA compressed {len(self.feature_names)} original features into "
            f"{components} summary features while keeping {variance:.0%} of the information."
        )
