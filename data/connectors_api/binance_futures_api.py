from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
import time

from data.utils.net_utils import get_json, iso_from_ms, timestamp_ms


BASE_URL = "https://fapi.binance.com"


def fetch_funding_rate_events(symbol, start_time, limit=1000):
    event_rows = []
    next_start_ms = timestamp_ms(start_time)

    while True:
        params = {
            "symbol": symbol,
            "startTime": next_start_ms,
            "limit": limit,
        }
        url = f"{BASE_URL}/fapi/v1/fundingRate?{urlencode(params)}"
        print(f"Fetching {symbol} funding rates from {iso_from_ms(next_start_ms)}...")

        try:
            batch = get_json(url)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Binance futures HTTP error for {symbol}: {exc.code} {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error while fetching {symbol} funding rates: {exc}") from exc

        if not batch:
            break

        for row in batch:
            funding_time_ms = int(row["fundingTime"])
            funding_timestamp = iso_from_ms(funding_time_ms)
            event_rows.append(
                {
                    "timestamp": funding_timestamp,
                    "date": funding_timestamp[:10],
                    "symbol": row["symbol"],
                    "funding_rate": row["fundingRate"],
                    "mark_price": row.get("markPrice", ""),
                }
            )

        if len(batch) < limit:
            break

        next_start_ms = int(batch[-1]["fundingTime"]) + 1
        time.sleep(0.25)

    event_rows = list({row["timestamp"]: row for row in event_rows}.values())
    return sorted(event_rows, key=lambda row: row["timestamp"])
