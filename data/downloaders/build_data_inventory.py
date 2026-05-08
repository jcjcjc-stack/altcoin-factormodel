from pathlib import Path
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.config.data_sources import DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR
from data.utils.io_utils import write_csv


inventory_path = DATA_DIR / "data_inventory.csv"

report_configs = [
    {
        "source": "cryptocompare",
        "dataset": "daily_ohlcv",
        "folder": RAW_DATA_DIR / "cryptocompare_daily",
        "name_column": "symbol",
        "rows_column": "rows",
        "first_column": "first_timestamp",
        "last_column": "last_timestamp",
        "combined_file": "cryptocompare_daily_ohlcv_combined.csv",
    },
    {
        "source": "fred",
        "dataset": "macro",
        "folder": RAW_DATA_DIR / "fred_macro",
        "name_column": "factor",
        "rows_column": "rows",
        "first_column": "first_date",
        "last_column": "last_date",
        "combined_file": "fred_macro_combined.csv",
    },
    {
        "source": "yfinance",
        "dataset": "market_ohlcv",
        "folder": RAW_DATA_DIR / "yfinance_market",
        "name_column": "factor",
        "rows_column": "rows",
        "first_column": "first_date",
        "last_column": "last_date",
        "combined_file": "yfinance_market_ohlcv_combined.csv",
    },
    {
        "source": "binance",
        "dataset": "btc_funding",
        "folder": RAW_DATA_DIR / "binance_funding",
        "name_column": "symbol",
        "rows_column": "daily_rows",
        "first_column": "first_timestamp",
        "last_column": "last_timestamp",
        "combined_file": "BTCUSDT_funding_rate_daily.csv",
    },
    {
        "source": "deribit",
        "dataset": "btc_dvol",
        "folder": RAW_DATA_DIR / "deribit_dvol",
        "name_column": "currency",
        "rows_column": "rows",
        "first_column": "first_timestamp",
        "last_column": "last_timestamp",
        "combined_file": "BTC_DVOL_1d.csv",
    },
]

inventory_rows = []
for config in report_configs:
    report_path = config["folder"] / "download_report.csv"
    if not report_path.exists():
        continue

    with report_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            inventory_rows.append(
                {
                    "source": config["source"],
                    "dataset": config["dataset"],
                    "name": row.get(config["name_column"], ""),
                    "rows": row.get(config["rows_column"], ""),
                    "first_date": row.get(config["first_column"], ""),
                    "last_date": row.get(config["last_column"], ""),
                    "raw_folder": str(config["folder"].relative_to(PROJECT_ROOT)),
                    "combined_path": str((config["folder"] / config["combined_file"]).relative_to(PROJECT_ROOT)),
                    "standard_combined_path": str((config["folder"] / "combined.csv").relative_to(PROJECT_ROOT)),
                }
            )

write_csv(
    inventory_path,
    inventory_rows,
    [
        "source",
        "dataset",
        "name",
        "rows",
        "first_date",
        "last_date",
        "raw_folder",
        "combined_path",
        "standard_combined_path",
    ],
)

print(f"Saved data inventory to {inventory_path}")
