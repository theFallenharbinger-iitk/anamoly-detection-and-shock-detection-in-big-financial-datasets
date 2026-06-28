from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticDataset:
    """Data plus the hidden answer key used only for testing the prototype."""

    data: pd.DataFrame
    event_log: pd.DataFrame


def make_synthetic_financial_data(
    days: int = 900,
    assets: int = 80,
    sectors: int = 8,
    random_state: int = 42,
) -> SyntheticDataset:
    """Create high-dimensional return data with a few planted unusual events.

    Each row is one day. Each column is one asset return.
    Most days are normal. A small number of days contain planted anomalies.
    """

    rng = np.random.default_rng(random_state)
    dates = pd.bdate_range("2021-01-01", periods=days)

    asset_names = [f"Asset_{i + 1:03d}" for i in range(assets)]
    sector_for_asset = np.array([i % sectors for i in range(assets)])

    market = rng.normal(0, 0.008, size=(days, 1))
    sector_factors = rng.normal(0, 0.006, size=(days, sectors))
    noise = rng.normal(0, 0.012, size=(days, assets))

    returns = 0.65 * market + 0.45 * sector_factors[:, sector_for_asset] + noise
    event_notes: list[dict[str, object]] = []

    planted_anomaly = np.zeros(days, dtype=bool)
    anomaly_type = np.full(days, "normal", dtype=object)

    def mark(day_index: int, event_type: str, plain_reason: str) -> None:
        planted_anomaly[day_index] = True
        anomaly_type[day_index] = event_type
        event_notes.append(
            {
                "date": dates[day_index],
                "event_type": event_type,
                "plain_reason": plain_reason,
            }
        )

    def day_at(fraction: float) -> int:
        return min(max(int(days * fraction), 0), days - 1)

    # A broad market crash: many assets move down together.
    crash_start = day_at(0.20)
    for day in range(crash_start, min(crash_start + 3, days)):
        returns[day, :] += rng.normal(-0.075, 0.012, size=assets)
        mark(day, "market_crash", "Almost every asset dropped on the same day.")

    # A sector shock: one group of related assets moves sharply.
    shock_sector = 3
    sector_assets = sector_for_asset == shock_sector
    sector_start = day_at(0.46)
    for day in range(sector_start, min(sector_start + 2, days)):
        returns[day, sector_assets] += rng.normal(0.09, 0.015, size=sector_assets.sum())
        mark(day, "sector_shock", "One sector moved much more than the rest of the market.")

    # A quiet-to-wild volatility burst: values are not all one direction, but the day is noisy.
    volatility_start = day_at(0.71)
    for day in range(volatility_start, min(volatility_start + 6, days)):
        returns[day, :] += rng.normal(0, 0.055, size=assets)
        mark(day, "volatility_burst", "Many assets became unusually jumpy at the same time.")

    # Single asset spikes: isolated weird behavior in one or two columns.
    spike_events = [
        (day_at(0.80), 5 % assets, 0.16),
        (day_at(0.84), 52 % assets, -0.18),
        (day_at(0.90), 24 % assets, 0.15),
    ]
    for day, asset_index, jump in spike_events:
        returns[day, asset_index] += jump
        mark(day, "single_asset_spike", "One asset moved far more than its usual daily range.")

    data = pd.DataFrame(returns, index=dates, columns=asset_names)
    data.index.name = "date"
    data["planted_anomaly"] = planted_anomaly
    data["planted_type"] = anomaly_type

    return SyntheticDataset(data=data, event_log=pd.DataFrame(event_notes))


def feature_groups(columns: list[str], sectors: int = 8) -> dict[str, str]:
    """Give each asset a simple group name so explanations can mention sectors."""

    groups: dict[str, str] = {}
    for i, column in enumerate(columns):
        groups[column] = f"Sector {i % sectors + 1}"
    return groups
