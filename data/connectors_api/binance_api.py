from __future__ import annotations

import time

import pandas as pd
import requests


INTERVAL_MS = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}


def get_binance_data(
    symbol: str,
    interval: str = "1m",
    start_time: object = None,
    end_time: object = None,
    limit: int = 1000,
    base_url: str = "https://data-api.binance.vision",
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """
    Pull Binance kline data and return only the fields needed for analysis.

    Output:
    datetime index, symbol, close, volume, trades, vol_bid, vol_ask

    Binance allows max 1000 candles per request. If start_time/end_time span
    more than 1000 candles, this function loops until the full range is loaded.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")

    symbol = symbol.upper()
    start_ms = None if start_time is None else int(pd.to_datetime(start_time, utc=True).timestamp() * 1000)
    end_ms = None if end_time is None else int(pd.to_datetime(end_time, utc=True).timestamp() * 1000)

    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        raise ValueError("start_time must be before end_time")

    rows = []
    next_start_ms = start_ms
    session = requests.Session()

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if next_start_ms is not None:
            params["startTime"] = next_start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        response = session.get(f"{base_url.rstrip('/')}/api/v3/klines", params=params, timeout=10)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        rows.extend(batch)

        if start_ms is None:
            break
        if len(batch) < limit:
            break

        next_start_ms = int(batch[-1][0]) + INTERVAL_MS[interval]
        if end_ms is not None and next_start_ms >= end_ms:
            break

        time.sleep(sleep_seconds)

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "vol_ask",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    if df.empty:
        empty = pd.DataFrame(columns=["symbol", "close", "volume", "trades", "vol_bid", "vol_ask"])
        empty.index.name = "datetime"
        return empty

    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])
    df["trades"] = pd.to_numeric(df["trades"], downcast="integer")
    df["vol_ask"] = pd.to_numeric(df["vol_ask"])
    df["vol_bid"] = df["volume"] - df["vol_ask"]

    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df = df.set_index("datetime")

    return df[["symbol", "close", "volume", "trades", "vol_bid", "vol_ask"]]


if __name__ == "__main__":
    btc = get_binance_data(
        symbol="BTCUSDT",
        interval="1m",
        start_time="2020-01-01 00:00:00",
        end_time="2024-01-02 00:00:00",
    )
    print(btc.head())
    print(btc.tail())
    print(f"Rows: {len(btc)}")
