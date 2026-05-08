from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.config.data_sources import (
    CRYPTOCOMPARE_COINS,
    CRYPTOCOMPARE_QUOTE,
    RAW_DATA_DIR,
    RESEARCH_START_TIME,
)
from data.connectors_api.cryptocompare_api import fetch_daily_candles
from data.utils.io_utils import write_csv, write_download_report
from data.utils.net_utils import timestamp_ms

START_TS = timestamp_ms(RESEARCH_START_TIME) // 1000
LIMIT = 2000
output_dir = RAW_DATA_DIR / "cryptocompare_daily"

candle_fields = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume_from",
    "volume_to",
]

combined_rows = []
report_rows = []

for coin in CRYPTOCOMPARE_COINS:
    print(f"Fetching {coin}/{CRYPTOCOMPARE_QUOTE} daily candles from CryptoCompare...")
    rows = fetch_daily_candles(coin, CRYPTOCOMPARE_QUOTE, START_TS, LIMIT)

    output_path = output_dir / f"{coin}{CRYPTOCOMPARE_QUOTE}_1d.csv"
    write_csv(output_path, rows, candle_fields)

    combined_rows.extend(rows)
    report_rows.append(
        {
            "symbol": f"{coin}{CRYPTOCOMPARE_QUOTE}",
            "rows": len(rows),
            "first_timestamp": rows[0]["timestamp"] if rows else "",
            "last_timestamp": rows[-1]["timestamp"] if rows else "",
        }
    )

    print(f"  saved {len(rows):,} rows to {output_path}")

combined_path = output_dir / "cryptocompare_daily_ohlcv_combined.csv"
write_csv(combined_path, combined_rows, candle_fields)
write_csv(output_dir / "combined.csv", combined_rows, candle_fields)

report_path = output_dir / "download_report.csv"
write_download_report(report_path, report_rows, ["symbol", "rows", "first_timestamp", "last_timestamp"])

print(f"Saved combined file to {combined_path}")
print(f"Saved download report to {report_path}")
for row in report_rows:
    print(row)
