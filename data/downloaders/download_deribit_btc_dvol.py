from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.config.data_sources import DERIBIT_DVOL_CURRENCY, RAW_DATA_DIR, RESEARCH_START_TIME, RESEARCH_TIME_INTERVAL
from data.connectors_api.deribit_api import fetch_dvol_candles
from data.utils.io_utils import write_csv, write_download_report
from data.utils.time_interval_utils import aggregate_rows


CURRENCY = DERIBIT_DVOL_CURRENCY
OUTPUT_INTERVAL = RESEARCH_TIME_INTERVAL
DERIBIT_RESOLUTIONS = {
    "1H": "3600",
    "1D": "1D",
    "1M": "1D",
    "1Y": "1D",
}
RESOLUTION = DERIBIT_RESOLUTIONS[OUTPUT_INTERVAL]
output_dir = RAW_DATA_DIR / "deribit_dvol"
dvol_fields = ["timestamp", "date", "currency", "open", "high", "low", "close"]

raw_rows = fetch_dvol_candles(CURRENCY, RESEARCH_START_TIME, RESOLUTION)
rows = aggregate_rows(
    raw_rows,
    OUTPUT_INTERVAL,
    "timestamp",
    group_columns=("currency",),
    first_columns=("date", "open"),
    max_columns=("high",),
    min_columns=("low",),
    last_columns=("close",),
    timestamp_output=True,
)
for row in rows:
    row["date"] = row["timestamp"][:10]

output_path = output_dir / f"{CURRENCY}_DVOL_{OUTPUT_INTERVAL}.csv"
write_csv(output_path, rows, dvol_fields)
write_csv(output_dir / "combined.csv", rows, dvol_fields)

report_path = output_dir / "download_report.csv"
write_download_report(
    report_path,
    [
        {
            "currency": CURRENCY,
            "rows": len(rows),
            "first_timestamp": rows[0]["timestamp"] if rows else "",
            "last_timestamp": rows[-1]["timestamp"] if rows else "",
        }
    ],
    ["currency", "rows", "first_timestamp", "last_timestamp"],
)

print(f"Saved DVOL candles to {output_path}")
print(f"Saved standard combined file to {output_dir / 'combined.csv'}")
print(f"Saved download report to {report_path}")
