from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.connectors_api.binance_futures_api import fetch_funding_rate_events
from data.config.data_sources import BINANCE_BTC_FUNDING_SYMBOL, RAW_DATA_DIR, RESEARCH_START_TIME, RESEARCH_TIME_INTERVAL
from data.utils.io_utils import write_csv, write_download_report
from data.utils.time_interval_utils import bucket_value

SYMBOL = BINANCE_BTC_FUNDING_SYMBOL
OUTPUT_INTERVAL = RESEARCH_TIME_INTERVAL
output_dir = RAW_DATA_DIR / "binance_funding"


event_rows = fetch_funding_rate_events(SYMBOL, RESEARCH_START_TIME)

daily_groups = {}
for row in event_rows:
    bucket_source = row["timestamp"] if OUTPUT_INTERVAL == "1H" else row["date"]
    bucket = bucket_value(bucket_source, OUTPUT_INTERVAL)
    daily_groups.setdefault(bucket, []).append(float(row["funding_rate"]))

daily_rows = [
    {
        "date": date,
        "symbol": SYMBOL,
        "funding_rate_sum": sum(values),
        "funding_rate_mean": sum(values) / len(values),
        "funding_events": len(values),
    }
    for date, values in sorted(daily_groups.items())
]

event_path = output_dir / f"{SYMBOL}_funding_rate_events.csv"
write_csv(event_path, event_rows, ["timestamp", "date", "symbol", "funding_rate", "mark_price"])

daily_path = output_dir / f"{SYMBOL}_funding_rate_{OUTPUT_INTERVAL}.csv"
write_csv(daily_path, daily_rows, ["date", "symbol", "funding_rate_sum", "funding_rate_mean", "funding_events"])
write_csv(output_dir / "combined.csv", daily_rows, ["date", "symbol", "funding_rate_sum", "funding_rate_mean", "funding_events"])

report_path = output_dir / "download_report.csv"
write_download_report(
    report_path,
    [
        {
            "symbol": SYMBOL,
            "event_rows": len(event_rows),
            "daily_rows": len(daily_rows),
            "first_timestamp": event_rows[0]["timestamp"] if event_rows else "",
            "last_timestamp": event_rows[-1]["timestamp"] if event_rows else "",
        }
    ],
    ["symbol", "event_rows", "daily_rows", "first_timestamp", "last_timestamp"],
)

print(f"Saved funding events to {event_path}")
print(f"Saved daily funding file to {daily_path}")
print(f"Saved download report to {report_path}")
