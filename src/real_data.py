from __future__ import annotations

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "JPM",
    "XOM",
    "UNH",
    "PG",
    "SPY",
    "QQQ",
]


def download_stock_returns(
    tickers: list[str] | None = None,
    start: str = "2020-01-01",
    end: str | None = None,
    timeout: int = 20,
) -> pd.DataFrame:
    """Download stock prices and convert them into daily returns.

    The detector should use returns, not raw prices.
    Raw prices trend over time. Returns describe daily movement, which is what
    anomaly detection usually needs.
    """

    selected_tickers = tickers or DEFAULT_TICKERS
    clean_tickers = [ticker.strip().upper() for ticker in selected_tickers if ticker.strip()]
    if not clean_tickers:
        raise ValueError("Please provide at least one ticker.")

    raw = yf.download(
        clean_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
        timeout=timeout,
    )

    if raw.empty:
        raise ValueError("No data was downloaded. Check the tickers and date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain close prices.")
        prices = raw["Close"]
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data does not contain close prices.")
        prices = raw[["Close"]]
        prices.columns = clean_tickers[:1]

    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    returns = prices.pct_change().dropna(how="all")
    returns = returns.dropna(axis=1, thresh=max(10, int(len(returns) * 0.80)))
    returns = returns.dropna()

    if returns.shape[0] < 50:
        raise ValueError("Not enough usable rows after cleaning. Try an earlier start date.")

    if returns.shape[1] < 2:
        raise ValueError("Need at least two usable tickers after cleaning.")

    returns.index.name = "date"
    return returns


def simple_feature_groups(columns: list[str]) -> dict[str, str]:
    """Group real tickers under one readable label for explanations."""

    return {column: "Real market ticker" for column in columns}


def load_market_csv(
    csv_path: str,
    date_column: str = "date",
    input_kind: str = "prices",
) -> pd.DataFrame:
    """Load a local CSV and return daily returns for the detector.

    Expected format:
    - one date column
    - every other numeric column is one ticker, asset, or signal

    If `input_kind` is "prices", the function converts prices to returns.
    If `input_kind` is "returns", the function uses the numeric values directly.
    """

    data = pd.read_csv(csv_path)
    if date_column not in data.columns:
        raise ValueError(f"CSV must contain a '{date_column}' column.")

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.set_index(date_column).sort_index()

    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("CSV needs at least two numeric feature columns.")

    if input_kind == "prices":
        returns = numeric.ffill().pct_change().dropna()
    elif input_kind == "returns":
        returns = numeric.dropna()
    else:
        raise ValueError("input_kind must be either 'prices' or 'returns'.")

    returns = returns.dropna(axis=1, thresh=max(10, int(len(returns) * 0.80))).dropna()
    if returns.shape[0] < 50:
        raise ValueError("Not enough usable rows after cleaning. Use a larger CSV.")

    returns.index.name = "date"
    return returns
