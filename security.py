"""Small, dependency-free security helpers for MerchantPulse AI."""
from __future__ import annotations

import hmac
import os
import re
import time
from typing import Any, Callable

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def mask_email(email: Any) -> str:
    """Show a buyer email as b***@example.com so the UI never exposes it in full."""
    if not isinstance(email, str) or "@" not in email:
        return "—"
    local, _, domain = email.partition("@")
    if not local or not domain:
        return "—"
    return f"{local[0]}***@{domain}"


def clean_text(value: Any, max_len: int) -> str:
    """Strip control characters, collapse whitespace and cap the length."""
    if not isinstance(value, str):
        return ""
    text = _CONTROL_CHARS.sub("", value)
    return " ".join(text.split())[:max_len]


def approval_pin_required() -> bool:
    """True when MERCHANT_APPROVAL_PIN is set, so refunds need a PIN."""
    return bool(os.getenv("MERCHANT_APPROVAL_PIN", "").strip())


def verify_approval_pin(entered: Any) -> bool:
    """Constant-time PIN check. With no PIN configured, approval is open (local demo)."""
    expected = os.getenv("MERCHANT_APPROVAL_PIN", "").strip()
    if not expected:
        return True
    if not isinstance(entered, str):
        return False
    return hmac.compare_digest(entered.strip().encode("utf-8"), expected.encode("utf-8"))


def allow_event(
    events: list[float],
    *,
    max_events: int,
    window_seconds: float,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Sliding-window limiter. `events` is caller-owned state (e.g. st.session_state)."""
    now = clock()
    events[:] = [t for t in events if now - t < window_seconds]
    if len(events) >= max_events:
        return False
    events.append(now)
    return True
