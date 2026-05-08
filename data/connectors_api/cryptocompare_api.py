import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from data.utils.net_utils import get_json


BASE_URL = "https://min-api.cryptocompare.com/data/v2/histoday"


def fetch_daily_candles(coin, quote, start_ts, limit=2000):
    params = {
        "fsym": coin,
        "tsym": quote,
        "limit": limit,
        "toTs": int(time.time()),
    }
    url = f"{BASE_URL}?{urlencode(params)}"

    try:
        payload = get_json(url)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"CryptoCompare HTTP error for {coin}: {exc.code} {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while fetching {coin}: {exc}") from exc

    if payload.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare returned an error for {coin}: {payload}")

    rows = []
    for candle in payload["Data"]["Data"]:
        if int(candle["time"]) < start_ts:
            continue

        rows.append(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(candle["time"])),
                "symbol": f"{coin}{quote}",
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume_from": candle["volumefrom"],
                "volume_to": candle["volumeto"],
            }
        )

    return rows
