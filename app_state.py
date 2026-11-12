"""Small, testable helpers for Streamlit session-state transitions."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

PENDING_ORDER_KEY = "paypal_pending_order"


def clear_pending_order(
    session_state: MutableMapping[str, Any],
    order_id: str | None = None,
) -> bool:
    """Clear the stored PayPal checkout only when it matches the completed/cancelled flow.

    Pass ``order_id`` after a confirmed capture to avoid clearing a different active
    checkout. Omit it after an explicit buyer cancellation, where the single pending
    checkout in this browser session should be cleared.
    """
    pending = session_state.get(PENDING_ORDER_KEY)
    if not isinstance(pending, dict):
        return False
    if order_id is not None and pending.get("order_id") != order_id:
        return False
    session_state.pop(PENDING_ORDER_KEY, None)
    return True
