from pathlib import Path
import subprocess
import sys


# Edit this section, then run:
# python refresh_data.py
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

RESEARCH_START_DATE = "2024-08-02"
RESEARCH_START_TIME = "2024-08-02 00:00:00+00:00"
RESEARCH_TIME_INTERVAL = "1H"  # Use "1H", "1D", "1M", or "1Y".
ALLOW_INSECURE_SSL_FALLBACK = True

BINANCE_SPOT_SYMBOLS = [
    "PEPEUSDT",
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "SHIBUSDT",
]

FRED_SERIES = {
    "M2SL": "m2_money_supply",
    "SOFR": "sofr",
    "DFF": "effective_fed_funds_rate",
}

YFINANCE_TICKERS = {
    "QQQ": "nasdaq_100_etf",
    "SPY": "sp500_etf",
    "IEF": "treasury_7_10y_etf",
    "TLT": "treasury_20y_etf",
    "UUP": "us_dollar_index_proxy",
    "HYG": "high_yield_credit_proxy",
    "VXX": "vix_futures_proxy",
}

BINANCE_BTC_FUNDING_SYMBOL = "BTCUSDT"
DERIBIT_DVOL_CURRENCY = "BTC"


DOWNLOADERS = [
    "data/downloaders/download_binance_daily_candles.py",
    "data/downloaders/download_fred_macro_data.py",
    "data/downloaders/download_yfinance_market_data.py",
    "data/downloaders/download_binance_btc_funding_rate.py",
    "data/downloaders/download_deribit_btc_dvol.py",
    "data/downloaders/build_data_inventory.py",
]


def main():
    print("Refreshing PEPE research data")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Start date: {RESEARCH_START_DATE}")
    print(f"Time interval: {RESEARCH_TIME_INTERVAL}")
    print(f"Binance spot symbols: {', '.join(BINANCE_SPOT_SYMBOLS)}")
    print(f"FRED series: {', '.join(FRED_SERIES)}")
    print(f"Yahoo Finance tickers: {', '.join(YFINANCE_TICKERS)}")
    print(f"Binance funding symbol: {BINANCE_BTC_FUNDING_SYMBOL}")
    print(f"Deribit DVOL currency: {DERIBIT_DVOL_CURRENCY}")
    print("Download steps:")
    for downloader in DOWNLOADERS:
        print(f"  - {downloader}")

    if any(arg in {"--show-config", "--dry-run"} for arg in sys.argv[1:]):
        print("\nDry run only. No data was downloaded.")
        return

    for downloader in DOWNLOADERS:
        script_path = PROJECT_ROOT / downloader
        print(f"\nRunning {downloader}...")
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)

    print("\nFinished refreshing research data.")
    print(f"Inventory: {DATA_DIR / 'data_inventory.csv'}")


if __name__ == "__main__":
    main()
