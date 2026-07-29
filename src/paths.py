from pathlib import Path


def find_project_root(start=None):
    current = Path.cwd().resolve() if start is None else Path(start).resolve()
    for path in [current, *current.parents]:
        if (path / "refresh_data.py").exists() and (path / "data").exists():
            return path
    raise FileNotFoundError("Could not find project root containing refresh_data.py and data/.")


def data_paths(project_root, time_interval):
    root = Path(project_root)
    return {
        "binance_spot": root / "data" / "raw" / "binance_daily" / f"binance_spot_{time_interval}_ohlcv_combined.csv",
        "fred_macro": root / "data" / "raw" / "fred_macro" / "fred_macro_1D_combined.csv",
        "yfinance_market": root
        / "data"
        / "raw"
        / "yfinance_market"
        / f"yfinance_market_{time_interval}_ohlcv_combined.csv",
        "btc_funding": root
        / "data"
        / "raw"
        / "binance_funding"
        / f"BTCUSDT_funding_rate_{time_interval}.csv",
        "btc_dvol": root / "data" / "raw" / "deribit_dvol" / f"BTC_DVOL_{time_interval}.csv",
    }

