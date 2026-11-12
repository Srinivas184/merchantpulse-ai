"""Tests for safe Streamlit checkout session-state transitions."""
from __future__ import annotations

import unittest

from app_state import PENDING_ORDER_KEY, clear_pending_order


class PendingOrderStateTests(unittest.TestCase):
    def test_confirmed_capture_clears_matching_pending_order(self):
        state = {PENDING_ORDER_KEY: {"order_id": "ORDER-123", "amount": "5.00"}}

        cleared = clear_pending_order(state, "ORDER-123")

        self.assertTrue(cleared)
        self.assertNotIn(PENDING_ORDER_KEY, state)

    def test_capture_does_not_clear_a_different_pending_order(self):
        pending = {"order_id": "ORDER-456", "amount": "15.00"}
        state = {PENDING_ORDER_KEY: pending}

        cleared = clear_pending_order(state, "ORDER-123")

        self.assertFalse(cleared)
        self.assertEqual(state[PENDING_ORDER_KEY], pending)

    def test_buyer_cancellation_clears_pending_order(self):
        state = {PENDING_ORDER_KEY: {"order_id": "ORDER-123"}}

        cleared = clear_pending_order(state)

        self.assertTrue(cleared)
        self.assertNotIn(PENDING_ORDER_KEY, state)

    def test_missing_or_malformed_pending_order_is_safe(self):
        self.assertFalse(clear_pending_order({}))
        state = {PENDING_ORDER_KEY: "not-a-record"}
        self.assertFalse(clear_pending_order(state, "ORDER-123"))
        self.assertEqual(state[PENDING_ORDER_KEY], "not-a-record")


if __name__ == "__main__":
    unittest.main(verbosity=2)
