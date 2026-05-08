from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import urlopen
import json
import ssl

from data.config.data_sources import ALLOW_INSECURE_SSL_FALLBACK


def get_json(url, timeout=30):
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        if not ALLOW_INSECURE_SSL_FALLBACK or "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise

        context = ssl._create_unverified_context()
        with urlopen(url, timeout=timeout, context=context) as response:
            return json.loads(response.read().decode("utf-8"))


def get_text(url, timeout=30):
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except URLError as exc:
        if not ALLOW_INSECURE_SSL_FALLBACK or "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise

        context = ssl._create_unverified_context()
        with urlopen(url, timeout=timeout, context=context) as response:
            return response.read().decode("utf-8")


def timestamp_ms(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def iso_from_ms(value):
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
