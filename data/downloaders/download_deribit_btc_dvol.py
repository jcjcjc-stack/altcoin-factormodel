from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.config.data_sources import DERIBIT_DVOL_CURRENCY, RAW_DATA_DIR, RESEARCH_START_TIME
from data.connectors_api.deribit_api import fetch_dvol_candles
from data.utils.io_utils import write_csv, write_download_report


CURRENCY = DERIBIT_DVOL_CURRENCY
RESOLUTION = "1D"
output_dir = RAW_DATA_DIR / "deribit_dvol"
dvol_fields = ["timestamp", "date", "currency", "open", "high", "low", "close"]

rows = fetch_dvol_candles(CURRENCY, RESEARCH_START_TIME, RESOLUTION)

output_path = output_dir / f"{CURRENCY}_DVOL_1d.csv"
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
