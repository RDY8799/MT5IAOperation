from __future__ import annotations

import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone


_LAST_SENT: dict[str, datetime] = {}


def _enabled() -> bool:
    return bool(os.getenv("MT5_TELEGRAM_BOT_TOKEN")) and bool(os.getenv("MT5_TELEGRAM_CHAT_ID"))


def _should_send(key: str, cooldown_seconds: int) -> bool:
    if cooldown_seconds <= 0:
        return True
    now = datetime.now(timezone.utc)
    last = _LAST_SENT.get(key)
    if last is None or now - last >= timedelta(seconds=cooldown_seconds):
        _LAST_SENT[key] = now
        return True
    return False


def send_telegram(message: str, key: str = "default", cooldown_seconds: int = 0) -> bool:
    if not _enabled():
        return False
    if not _should_send(key=key, cooldown_seconds=cooldown_seconds):
        return False

    token = os.getenv("MT5_TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("MT5_TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False

    base_url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(base_url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except Exception:
        return False

