from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import urlopen
import json
import socket
import ssl
import time

from data.config.data_sources import ALLOW_INSECURE_SSL_FALLBACK


def get_json(url, timeout=45, retries=3, retry_delay=2):
    return json.loads(get_text(url, timeout=timeout, retries=retries, retry_delay=retry_delay))


def get_text(url, timeout=45, retries=3, retry_delay=2):
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except URLError as exc:
            if ALLOW_INSECURE_SSL_FALLBACK and "CERTIFICATE_VERIFY_FAILED" in str(exc):
                context = ssl._create_unverified_context()
                with urlopen(url, timeout=timeout, context=context) as response:
                    return response.read().decode("utf-8")
            last_exc = exc
        except (TimeoutError, socket.timeout) as exc:
            last_exc = exc

        if attempt < retries:
            print(f"Request failed on attempt {attempt}/{retries}; retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    raise last_exc


def timestamp_ms(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def iso_from_ms(value):
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
