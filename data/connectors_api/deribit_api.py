from datetime import datetime, timezone, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
import time

from data.utils.net_utils import get_json, iso_from_ms


BASE_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def fetch_dvol_candles(currency, start_time, resolution="1D", chunk_days=900):
    rows = []
    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end_dt = datetime.now(timezone.utc)

    while start_dt < end_dt:
        chunk_end_dt = min(start_dt + timedelta(days=chunk_days), end_dt)
        params = {
            "currency": currency,
            "start_timestamp": int(start_dt.timestamp() * 1000),
            "end_timestamp": int(chunk_end_dt.timestamp() * 1000),
            "resolution": resolution,
        }
        url = f"{BASE_URL}?{urlencode(params)}"

        print(f"Fetching Deribit {currency} DVOL {resolution} candles from {start_dt.isoformat()}...")
        try:
            payload = get_json(url)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Deribit HTTP error for {currency} DVOL: {exc.code} {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error while fetching {currency} DVOL: {exc}") from exc

        if "error" in payload:
            raise RuntimeError(f"Deribit returned an error: {payload['error']}")

        for candle in payload.get("result", {}).get("data", []):
            timestamp = int(candle[0])
            rows.append(
                {
                    "timestamp": iso_from_ms(timestamp),
                    "date": iso_from_ms(timestamp)[:10],
                    "currency": currency,
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                }
            )

        start_dt = chunk_end_dt + timedelta(days=1)
        time.sleep(0.25)

    rows = list({row["timestamp"]: row for row in rows}.values())
    return sorted(rows, key=lambda row: row["timestamp"])
