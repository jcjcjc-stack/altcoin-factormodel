from datetime import datetime, timezone
from urllib.error import URLError
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json
import socket
import ssl
import time

from data.config.data_sources import ALLOW_INSECURE_SSL_FALLBACK


def get_json(url, timeout=45, retries=3, retry_delay=2):
    return json.loads(get_text(url, timeout=timeout, retries=retries, retry_delay=retry_delay))


def get_text(url, timeout=45, retries=3, retry_delay=2):
    last_exc = None
    request = Request(
        url,
        headers={
            "User-Agent": "Altcoin-FactorModel/1.0 (+https://github.com; research data downloader)",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            last_exc = exc
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
        except URLError as exc:
            if ALLOW_INSECURE_SSL_FALLBACK and "CERTIFICATE_VERIFY_FAILED" in str(exc):
                context = ssl._create_unverified_context()
                with urlopen(request, timeout=timeout, context=context) as response:
                    return response.read().decode("utf-8")
            last_exc = exc
        except (TimeoutError, socket.timeout, OSError) as exc:
            last_exc = exc

        if attempt < retries:
            print(
                f"Request failed on attempt {attempt}/{retries}: "
                f"{type(last_exc).__name__}: {last_exc}; retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)

    raise last_exc


def timestamp_ms(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def iso_from_ms(value):
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
