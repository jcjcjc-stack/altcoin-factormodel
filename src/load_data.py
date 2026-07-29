import numpy as np
import pandas as pd

from paths import data_paths


def load_market_data(project_root, time_interval, coin_order):
    paths = data_paths(project_root, time_interval)

    df = pd.read_csv(paths["binance_spot"], parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["coin"] = df["symbol"].str.replace("USDT", "", regex=False)

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    prices = {
        column: df.pivot(index="timestamp", columns="coin", values=column).sort_index()
        for column in ["open", "high", "low", "close", "volume"]
    }
    available_coins = set(prices["close"].columns)
    missing_coins = [coin for coin in coin_order if coin not in available_coins]
    if missing_coins:
        missing_symbols = [f"{coin}USDT" for coin in missing_coins]
        raise ValueError(
            "Missing Binance spot data for: "
            + ", ".join(missing_symbols)
            + ". Run python refresh_data.py, then rerun the notebook."
        )

    prices = {
        column: values[coin_order].replace([np.inf, -np.inf], np.nan)
        for column, values in prices.items()
    }
    close_prices = prices["close"]

    raw_returns = close_prices.pct_change(fill_method=None)
    raw_returns = raw_returns[coin_order].replace([np.inf, -np.inf], np.nan)
    log_returns = np.log(close_prices / close_prices.shift(1))
    log_returns = log_returns[coin_order].replace([np.inf, -np.inf], np.nan)

    for coin in coin_order:
        first_valid_timestamp = close_prices[coin].first_valid_index()
        if first_valid_timestamp is not None:
            raw_returns.loc[first_valid_timestamp, coin] = 0.0
            log_returns.loc[first_valid_timestamp, coin] = 0.0

    fred_data = pd.read_csv(paths["fred_macro"], parse_dates=["date"])
    fred_data["date"] = pd.to_datetime(fred_data["date"], utc=True)
    fred_data["value"] = pd.to_numeric(fred_data["value"], errors="coerce")
    fred_features = fred_data.pivot(index="date", columns="factor", values="value").sort_index()
    fred_features = fred_features.reindex(close_prices.index).ffill().replace([np.inf, -np.inf], np.nan)

    yfinance_data = pd.read_csv(paths["yfinance_market"], parse_dates=["date"])
    yfinance_data["date"] = pd.to_datetime(yfinance_data["date"], utc=True).dt.floor("h")
    yfinance_data["close"] = pd.to_numeric(yfinance_data["close"], errors="coerce")
    yfinance_closes = yfinance_data.pivot(index="date", columns="factor", values="close").sort_index()
    yfinance_closes = yfinance_closes.reindex(close_prices.index).ffill().replace([np.inf, -np.inf], np.nan)
    yfinance_log_returns = np.log(yfinance_closes / yfinance_closes.shift(1)).replace([np.inf, -np.inf], np.nan)

    btc_funding = pd.read_csv(paths["btc_funding"], parse_dates=["date"])
    btc_funding["date"] = pd.to_datetime(btc_funding["date"], utc=True)
    btc_funding["funding_rate_sum"] = pd.to_numeric(btc_funding["funding_rate_sum"], errors="coerce")
    btc_funding = btc_funding.set_index("date").sort_index()
    btc_funding = btc_funding.reindex(close_prices.index).ffill().replace([np.inf, -np.inf], np.nan)

    btc_dvol = pd.read_csv(paths["btc_dvol"])
    dvol_time_column = "timestamp" if "timestamp" in btc_dvol.columns else "date"
    btc_dvol[dvol_time_column] = pd.to_datetime(btc_dvol[dvol_time_column], utc=True)
    btc_dvol["close"] = pd.to_numeric(btc_dvol["close"], errors="coerce")
    btc_dvol = btc_dvol.set_index(dvol_time_column).sort_index()
    btc_dvol = btc_dvol.reindex(close_prices.index).ffill().replace([np.inf, -np.inf], np.nan)

    return {
        "open_prices": prices["open"],
        "high_prices": prices["high"],
        "low_prices": prices["low"],
        "close_prices": close_prices,
        "volume_from": prices["volume"],
        "raw_returns": raw_returns,
        "log_returns": log_returns,
        "fred_features": fred_features,
        "yfinance_log_returns": yfinance_log_returns,
        "btc_funding": btc_funding,
        "btc_dvol": btc_dvol,
        "cumulative_returns": np.exp(log_returns.cumsum()) - 1,
        "simple_cumulative_returns": (1 + raw_returns).cumprod() - 1,
    }
