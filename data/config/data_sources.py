from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

RESEARCH_START_DATE = "2023-05-01"
RESEARCH_START_TIME = "2023-05-01 00:00:00+00:00"
ALLOW_INSECURE_SSL_FALLBACK = True

CRYPTOCOMPARE_COINS = ["PEPE", "BTC", "ETH", "SOL", "DOGE", "SHIB"]
CRYPTOCOMPARE_QUOTE = "USD"

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
    "DGS10": "us_10y_treasury_yield",
    "VIXCLS": "vix_close",
    "DFF": "effective_fed_funds_rate",
    "DTWEXBGS": "broad_us_dollar_index",
    "BAMLH0A0HYM2": "high_yield_spread",
}

YFINANCE_TICKERS = {
    "QQQ": "nasdaq_100_etf",
    "SPY": "sp500_etf",
    "GLD": "gold_etf",
    "SLV": "silver_etf",
}

BINANCE_BTC_FUNDING_SYMBOL = "BTCUSDT"
DERIBIT_DVOL_CURRENCY = "BTC"
