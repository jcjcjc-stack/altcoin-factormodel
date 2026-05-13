from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
import csv
import socket

from data.utils.net_utils import get_text


BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id, factor_name, start_date):
    params = {"id": series_id, "observation_start": start_date}
    url = f"{BASE_URL}?{urlencode(params)}"

    try:
        text = get_text(url)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FRED HTTP error for {series_id}: {exc.code} {body}") from exc
    except (URLError, TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Network error while fetching {series_id}: {exc}") from exc

    rows = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        if row["observation_date"] < start_date:
            continue

        value = row.get(series_id, "")
        if not value or value == ".":
            continue

        rows.append(
            {
                "date": row["observation_date"],
                "series_id": series_id,
                "factor": factor_name,
                "value": value,
            }
        )

    return rows
