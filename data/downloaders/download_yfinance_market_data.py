from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.connectors_api.yfinance_api import fetch_daily_market_rows
from data.config.data_sources import RAW_DATA_DIR, RESEARCH_START_DATE, RESEARCH_TIME_INTERVAL, YFINANCE_TICKERS
from data.utils.io_utils import write_csv, write_download_report
from data.utils.time_interval_utils import aggregate_rows


output_dir = RAW_DATA_DIR / "yfinance_market"
output_interval = RESEARCH_TIME_INTERVAL
yfinance_interval = "1h" if output_interval == "1H" else "1d"
market_fields = [
    "date",
    "ticker",
    "factor",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

combined_rows = []
report_rows = []

for ticker, factor_name in YFINANCE_TICKERS.items():
    print(f"Fetching Yahoo Finance ticker {ticker} as {factor_name} at {yfinance_interval}...")
    raw_rows = fetch_daily_market_rows(ticker, factor_name, RESEARCH_START_DATE, yfinance_interval)
    rows = aggregate_rows(
        raw_rows,
        output_interval,
        "date",
        group_columns=("ticker", "factor"),
        first_columns=("open",),
        max_columns=("high",),
        min_columns=("low",),
        last_columns=("close", "adj_close"),
        sum_columns=("volume",),
    )
    combined_rows.extend(rows)

    output_path = output_dir / f"{factor_name}_{output_interval}.csv"
    write_csv(output_path, rows, market_fields)

    report_rows.append(
        {
            "ticker": ticker,
            "factor": factor_name,
            "rows": len(rows),
            "first_date": rows[0]["date"] if rows else "",
            "last_date": rows[-1]["date"] if rows else "",
        }
    )

    print(f"  saved {len(rows):,} rows to {output_path}")

combined_path = output_dir / f"yfinance_market_{output_interval}_ohlcv_combined.csv"
write_csv(combined_path, combined_rows, market_fields)
write_csv(output_dir / "combined.csv", combined_rows, market_fields)

report_path = output_dir / "download_report.csv"
write_download_report(report_path, report_rows, ["ticker", "factor", "rows", "first_date", "last_date"])

print(f"Saved combined file to {combined_path}")
print(f"Saved download report to {report_path}")
for row in report_rows:
    print(row)
