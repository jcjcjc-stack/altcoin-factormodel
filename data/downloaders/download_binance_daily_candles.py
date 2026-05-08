from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.config.data_sources import BINANCE_SPOT_SYMBOLS, RAW_DATA_DIR, RESEARCH_START_TIME
from data.utils.io_utils import write_csv
from data.utils.net_utils import get_json


BASE_URL = "https://data-api.binance.vision"
INTERVAL = "1d"
INTERVAL_MS = 86_400_000
LIMIT = 1000
START_TIME = RESEARCH_START_TIME
OUTPUT_DIR = RAW_DATA_DIR / "binance_daily"
SYMBOLS = BINANCE_SPOT_SYMBOLS


def fetch_daily_klines(
    symbol: str,
    start_time: str,
    end_time: str | None = None,
    sleep_seconds: float = 0.1,
) -> list[dict[str, object]]:
    """Fetch Binance spot daily klines, looping because Binance caps requests."""
    start_ms = timestamp_ms(start_time)
    end_ms = None if end_time is None else timestamp_ms(end_time)

    rows = []
    next_start_ms = start_ms

    while True:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": LIMIT,
            "startTime": next_start_ms,
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        url = f"{BASE_URL}/api/v3/klines?{urlencode(params)}"
        try:
            batch = get_json(url)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Binance HTTP error for {symbol}: {exc.code} {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error while fetching {symbol}: {exc}") from exc

        if not batch:
            break

        rows.extend(format_kline(symbol, row) for row in batch)

        if len(batch) < LIMIT:
            break

        next_start_ms = int(batch[-1][0]) + INTERVAL_MS
        if end_ms is not None and next_start_ms >= end_ms:
            break

        time.sleep(sleep_seconds)

    unique_rows = {row["timestamp"]: row for row in rows}
    return [unique_rows[key] for key in sorted(unique_rows)]


def timestamp_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def format_kline(symbol: str, row: list[object]) -> dict[str, object]:
    return {
        "timestamp": iso_from_ms(int(row[0])),
        "close_timestamp": iso_from_ms(int(row[6])),
        "symbol": symbol,
        "open": row[1],
        "high": row[2],
        "low": row[3],
        "close": row[4],
        "volume": row[5],
        "quote_asset_volume": row[7],
        "number_of_trades": row[8],
        "taker_buy_base_volume": row[9],
        "taker_buy_quote_volume": row[10],
    }


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_report(frames: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows = []
    expected_start = parse_iso(START_TIME)

    for symbol, data in frames.items():
        if not data:
            rows.append(
                {
                    "symbol": symbol,
                    "rows": 0,
                    "first_timestamp": "",
                    "last_timestamp": "",
                    "duplicate_timestamps": 0,
                    "missing_daily_gaps": "",
                    "note": "No data returned.",
                }
            )
            continue

        timestamps = [parse_iso(str(row["timestamp"])) for row in data]
        missing_gaps = sum(
            1
            for previous, current in zip(timestamps, timestamps[1:])
            if (current - previous).days > 1
        )
        first_timestamp = timestamps[0]
        duplicate_count = len(timestamps) - len(set(timestamps))

        rows.append(
            {
                "symbol": symbol,
                "rows": len(data),
                "first_timestamp": first_timestamp.isoformat(),
                "last_timestamp": timestamps[-1].isoformat(),
                "duplicate_timestamps": duplicate_count,
                "missing_daily_gaps": missing_gaps,
                "note": ""
                if first_timestamp <= expected_start
                else "Data starts after requested start, likely Binance listing date.",
            }
        )

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candle_fields = [
        "timestamp",
        "close_timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    report_fields = [
        "symbol",
        "rows",
        "first_timestamp",
        "last_timestamp",
        "duplicate_timestamps",
        "missing_daily_gaps",
        "note",
    ]

    frames = {}
    for symbol in SYMBOLS:
        print(f"Fetching {symbol} daily candles from {START_TIME}...")
        data = fetch_daily_klines(symbol=symbol, start_time=START_TIME)
        frames[symbol] = data
        output_path = OUTPUT_DIR / f"{symbol}_1d.csv"
        write_csv(output_path, data, candle_fields)
        print(f"  saved {len(data):,} rows to {output_path}")

    combined = [row for data in frames.values() for row in data]
    combined_path = OUTPUT_DIR / "binance_spot_daily_ohlcv_combined.csv"
    write_csv(combined_path, combined, candle_fields)

    report = build_report(frames)
    report_path = OUTPUT_DIR / "download_report.csv"
    write_csv(report_path, report, report_fields)

    print(f"Saved combined file to {combined_path}")
    print(f"Saved download report to {report_path}")
    for row in report:
        print(row)


if __name__ == "__main__":
    main()
