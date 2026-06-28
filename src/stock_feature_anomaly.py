from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class StockFeatureAnomalyResult:
    scored_data: pd.DataFrame
    top_anomalies: pd.DataFrame
    feature_columns: list[str]


class StockFeatureAnomalyAnalyzer:
    """Find price anomaly days and explain which features dominated them.

    This is for a single stock. The input can include price, volume, and any
    extra numeric features such as market return, sector return, sentiment,
    volatility, technical indicators, or fundamentals.
    """

    def __init__(
        self,
        contamination: float = 0.03,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=250,
            contamination=contamination,
            random_state=random_state,
        )
        self.feature_columns: list[str] = []

    def fit_predict(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
        volume_column: str | None = "volume",
    ) -> StockFeatureAnomalyResult:
        prepared = build_stock_features(
            data=data,
            price_column=price_column,
            volume_column=volume_column,
        )

        self.feature_columns = [
            column
            for column in prepared.columns
            if column not in {"price", "price_return", "price_anomaly_score"}
        ]

        x = prepared[self.feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
        prepared = prepared.loc[x.index].copy()
        x_scaled = self.scaler.fit_transform(x)

        anomaly_score = -self.model.fit(x_scaled).score_samples(x_scaled)
        severity = pd.Series(anomaly_score, index=prepared.index).rank(pct=True)
        flags = self.model.predict(x_scaled) == -1

        dominance = self._feature_dominance(x_scaled)
        dominance_text = [
            self._explain_feature_dominance(dominance[row_index], bool(flags[row_index]))
            for row_index in range(len(prepared))
        ]

        scored = prepared.copy()
        scored["anomaly_score"] = anomaly_score
        scored["severity_percentile"] = severity
        scored["model_flag"] = flags
        scored["dominant_features"] = dominance_text

        for feature_index, feature_name in enumerate(self.feature_columns):
            scored[f"contribution_{feature_name}"] = dominance[:, feature_index]

        top = scored.sort_values("anomaly_score", ascending=False).head(20)
        return StockFeatureAnomalyResult(
            scored_data=scored,
            top_anomalies=top,
            feature_columns=self.feature_columns,
        )

    def _feature_dominance(self, x_scaled: np.ndarray) -> np.ndarray:
        """Convert standardized feature deviations into percent contributions."""

        squared = x_scaled**2
        row_totals = squared.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = 1.0
        return squared / row_totals

    def _explain_feature_dominance(self, dominance_row: np.ndarray, flagged: bool) -> str:
        top_indices = np.argsort(dominance_row)[-5:][::-1]
        parts = [
            f"{self.feature_columns[i]} ({dominance_row[i]:.0%})"
            for i in top_indices
        ]
        prefix = "Flagged as a price-feature anomaly." if flagged else "Not flagged, but strongest feature signals were measured."
        return f"{prefix} Dominant features: {', '.join(parts)}."


def build_stock_features(
    data: pd.DataFrame,
    price_column: str = "close",
    volume_column: str | None = "volume",
) -> pd.DataFrame:
    """Build explainable stock features from price, volume, and extra columns."""

    normalized_columns = {column.lower(): column for column in data.columns}
    actual_price_column = normalized_columns.get(price_column.lower(), price_column)
    if actual_price_column not in data.columns:
        raise ValueError(f"Could not find price column '{price_column}'.")

    prepared = pd.DataFrame(index=data.index)
    price = data[actual_price_column].astype(float)
    prepared["price"] = price
    prepared["price_return"] = price.pct_change()
    prepared["absolute_return"] = prepared["price_return"].abs()
    prepared["rolling_5d_return"] = price.pct_change(5)
    prepared["rolling_20d_return"] = price.pct_change(20)
    prepared["rolling_10d_volatility"] = prepared["price_return"].rolling(10).std()
    prepared["rolling_30d_volatility"] = prepared["price_return"].rolling(30).std()
    prepared["moving_average_gap_10d"] = price / price.rolling(10).mean() - 1
    prepared["moving_average_gap_30d"] = price / price.rolling(30).mean() - 1

    if volume_column:
        actual_volume_column = normalized_columns.get(volume_column.lower(), volume_column)
        if actual_volume_column in data.columns:
            volume = data[actual_volume_column].astype(float)
            prepared["volume_change"] = volume.pct_change()
            prepared["volume_vs_20d_average"] = volume / volume.rolling(20).mean() - 1

    ignored = {actual_price_column}
    if volume_column:
        ignored.add(normalized_columns.get(volume_column.lower(), volume_column))

    extra_numeric = data.drop(columns=[column for column in ignored if column in data.columns])
    extra_numeric = extra_numeric.select_dtypes(include="number")
    for column in extra_numeric.columns:
        prepared[f"input_{column}"] = extra_numeric[column]

    return prepared.replace([np.inf, -np.inf], np.nan).dropna()


def load_single_stock_csv(
    csv_path: str,
    date_column: str = "date",
) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    if date_column not in data.columns:
        raise ValueError(f"CSV must contain a '{date_column}' column.")

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.set_index(date_column).sort_index()
    data.index.name = "date"
    return data
