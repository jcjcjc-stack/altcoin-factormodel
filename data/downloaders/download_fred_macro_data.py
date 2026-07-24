from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.connectors_api.fred_api import fetch_fred_series
from data.config.data_sources import FRED_SERIES, RAW_DATA_DIR, RESEARCH_START_DATE, RESEARCH_TIME_INTERVAL
from data.utils.io_utils import write_csv, write_download_report
from data.utils.time_interval_utils import aggregate_rows


output_dir = RAW_DATA_DIR / "fred_macro"
output_interval = "1D" if RESEARCH_TIME_INTERVAL == "1H" else RESEARCH_TIME_INTERVAL

combined_rows = []
report_rows = []

for series_id, factor_name in FRED_SERIES.items():
    print(f"Fetching FRED series {series_id} as {factor_name}...")
    raw_rows = fetch_fred_series(series_id, factor_name, RESEARCH_START_DATE)
    rows = aggregate_rows(
        raw_rows,
        output_interval,
        "date",
        group_columns=("series_id", "factor"),
        last_columns=("value",),
    )
    combined_rows.extend(rows)

    output_path = output_dir / f"{factor_name}_{output_interval}.csv"
    write_csv(output_path, rows, ["date", "series_id", "factor", "value"])

    report_rows.append(
        {
            "series_id": series_id,
            "factor": factor_name,
            "rows": len(rows),
            "first_date": rows[0]["date"] if rows else "",
            "last_date": rows[-1]["date"] if rows else "",
        }
    )

    print(f"  saved {len(rows):,} rows to {output_path}")
    time.sleep(0.25)

combined_path = output_dir / f"fred_macro_{output_interval}_combined.csv"
write_csv(combined_path, combined_rows, ["date", "series_id", "factor", "value"])
write_csv(output_dir / "combined.csv", combined_rows, ["date", "series_id", "factor", "value"])

report_path = output_dir / "download_report.csv"
write_download_report(report_path, report_rows, ["series_id", "factor", "rows", "first_date", "last_date"])

print(f"Saved combined file to {combined_path}")
print(f"Saved download report to {report_path}")
for row in report_rows:
    print(row)
