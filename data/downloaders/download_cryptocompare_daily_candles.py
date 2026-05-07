from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
import csv
import json
import ssl
import time


BASE_URL = "https://min-api.cryptocompare.com/data/v2/histoday"
START_TS = 1682899200  # 2023-05-01 00:00:00 UTC
LIMIT = 2000
TSYM = "USD"
ALLOW_INSECURE_SSL_FALLBACK = True

coins = ["PEPE", "BTC", "ETH", "SOL", "DOGE", "SHIB"]

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent
output_dir = data_dir / "raw" / "cryptocompare_daily"
output_dir.mkdir(parents=True, exist_ok=True)

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

for coin in coins:
    params = {
        "fsym": coin,
        "tsym": TSYM,
        "limit": LIMIT,
        "toTs": int(time.time()),
    }
    url = f"{BASE_URL}?{urlencode(params)}"

    print(f"Fetching {coin}/{TSYM} daily candles from CryptoCompare...")

    try:
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"CryptoCompare HTTP error for {coin}: {exc.code} {body}") from exc
    except URLError as exc:
        if not ALLOW_INSECURE_SSL_FALLBACK or "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise RuntimeError(f"Network error while fetching {coin}: {exc}") from exc

        context = ssl._create_unverified_context()
        with urlopen(url, timeout=30, context=context) as response:
            payload = json.loads(response.read().decode("utf-8"))

    if payload.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare returned an error for {coin}: {payload}")

    rows = []
    for candle in payload["Data"]["Data"]:
        if int(candle["time"]) < START_TS:
            continue

        rows.append(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(candle["time"])),
                "symbol": f"{coin}{TSYM}",
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume_from": candle["volumefrom"],
                "volume_to": candle["volumeto"],
            }
        )

    output_path = output_dir / f"{coin}{TSYM}_1d.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=candle_fields)
        writer.writeheader()
        writer.writerows(rows)

    combined_rows.extend(rows)
    report_rows.append(
        {
            "symbol": f"{coin}{TSYM}",
            "rows": len(rows),
            "first_timestamp": rows[0]["timestamp"] if rows else "",
            "last_timestamp": rows[-1]["timestamp"] if rows else "",
        }
    )

    print(f"  saved {len(rows):,} rows to {output_path}")
    time.sleep(0.25)

combined_path = output_dir / "cryptocompare_daily_ohlcv_combined.csv"
with combined_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=candle_fields)
    writer.writeheader()
    writer.writerows(combined_rows)

report_path = output_dir / "download_report.csv"
with report_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["symbol", "rows", "first_timestamp", "last_timestamp"])
    writer.writeheader()
    writer.writerows(report_rows)

print(f"Saved combined file to {combined_path}")
print(f"Saved download report to {report_path}")
for row in report_rows:
    print(row)
