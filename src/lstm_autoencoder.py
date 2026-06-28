from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LSTMDetectionResult:
    scored_windows: pd.DataFrame
    top_anomalies: pd.DataFrame
    training_losses: list[float]


class LSTMAutoencoderNet(nn.Module):
    """Small LSTM autoencoder for reconstructing windows of time-series data."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 12) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = torch.relu(self.to_latent(hidden[-1]))
        repeated = self.from_latent(latent).unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


class LSTMAutoencoderDetector:
    """Detect sequence anomalies using reconstruction error.

    The model learns to rebuild normal windows of data. If a window is hard to
    rebuild, the model gives it a high anomaly score.
    """

    def __init__(
        self,
        window_size: int = 20,
        hidden_dim: int = 32,
        latent_dim: int = 12,
        epochs: int = 25,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        contamination: float = 0.03,
        random_state: int = 42,
    ) -> None:
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.model: LSTMAutoencoderNet | None = None
        self.training_losses: list[float] = []

    def fit_predict(self, data: pd.DataFrame) -> LSTMDetectionResult:
        self.feature_names = [
            column
            for column in data.columns
            if column not in {"planted_anomaly", "planted_type"}
        ]
        if len(data) <= self.window_size:
            raise ValueError("Data must contain more rows than the LSTM window size.")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x = data[self.feature_names].astype(float)
        x_scaled = self.scaler.fit_transform(x)
        sequences = self._make_sequences(x_scaled)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sequence_tensor = torch.tensor(sequences, dtype=torch.float32)
        dataset = TensorDataset(sequence_tensor, sequence_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = LSTMAutoencoderNet(
            input_dim=len(self.feature_names),
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.training_losses = []

        self.model.train()
        for _ in range(self.epochs):
            epoch_losses: list[float] = []
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = loss_fn(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.training_losses.append(float(np.mean(epoch_losses)))

        scores, feature_errors = self._score_sequences(sequence_tensor, device)
        severity = pd.Series(scores).rank(pct=True).to_numpy()
        threshold = np.quantile(scores, 1 - self.contamination)
        flags = scores >= threshold

        end_dates = data.index[self.window_size - 1 :]
        scored = pd.DataFrame(index=end_dates)
        scored.index.name = data.index.name or "date"
        scored["lstm_reconstruction_error"] = scores
        scored["severity_percentile"] = severity
        scored["model_flag"] = flags
        scored["plain_explanation"] = [
            self._explain_window(feature_errors[i], bool(flags[i]))
            for i in range(len(scored))
        ]

        if "planted_anomaly" in data.columns:
            planted_flags = []
            planted_types = []
            planted_values = data["planted_anomaly"].astype(bool).to_numpy()
            type_values = data.get("planted_type", pd.Series(["unknown"] * len(data), index=data.index)).to_numpy()
            for start in range(len(scored)):
                end = start + self.window_size
                window_flags = planted_values[start:end]
                window_types = sorted(set(type_values[start:end][window_flags]))
                planted_flags.append(bool(window_flags.any()))
                planted_types.append(", ".join(window_types) if window_types else "normal")
            scored["planted_anomaly_in_window"] = planted_flags
            scored["planted_type_in_window"] = planted_types

        top = scored.sort_values("lstm_reconstruction_error", ascending=False).head(15)
        return LSTMDetectionResult(
            scored_windows=scored,
            top_anomalies=top,
            training_losses=self.training_losses,
        )

    def _make_sequences(self, x_scaled: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                x_scaled[start : start + self.window_size]
                for start in range(len(x_scaled) - self.window_size + 1)
            ]
        )

    def _score_sequences(
        self,
        sequence_tensor: torch.Tensor,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        self.model.eval()
        with torch.no_grad():
            x = sequence_tensor.to(device)
            reconstructed = self.model(x)
            errors = (x - reconstructed) ** 2
            scores = errors.mean(dim=(1, 2)).cpu().numpy()
            feature_errors = errors.mean(dim=1).cpu().numpy()
        return scores, feature_errors

    def _explain_window(self, feature_error: np.ndarray, flagged: bool) -> str:
        top_indices = np.argsort(feature_error)[-5:][::-1]
        total_error = float(feature_error.sum())
        if total_error <= 0:
            total_error = 1.0

        contributors = [
            f"{self.feature_names[i]} ({feature_error[i] / total_error:.0%} of reconstruction error)"
            for i in top_indices
        ]

        top_share = float(feature_error[top_indices[:2]].sum() / total_error)
        if top_share > 0.65:
            pattern = "This looks like a focused sequence anomaly because a small number of features explain most of the reconstruction error."
        else:
            pattern = "This looks like a broad sequence anomaly because the reconstruction error is spread across several features."

        prefix = "Flagged as unusual." if flagged else "Not flagged, but this window still has a measurable reconstruction error."
        return f"{prefix} {pattern} Biggest contributors: {', '.join(contributors)}."
