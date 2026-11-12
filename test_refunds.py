"""Offline tests for the human-approved Sandbox refund engine."""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import paypal_integration as pp


def _resp(status_code: int, payload: dict) -> Mock:
    obj = Mock()
    obj.status_code = status_code
    obj.json.return_value = payload
    return obj


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(timespec="seconds")


class RefundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = pp.DATABASE_PATH
        pp.DATABASE_PATH = Path(self.temp_dir.name) / "refunds.sqlite3"
        self.env_patch = patch.dict(
            "os.environ",
            {
                "PAYPAL_CLIENT_ID": "unit-test-client-id",
                "PAYPAL_CLIENT_SECRET": "unit-test-client-secret",
                "PAYPAL_BASE_URL": "https://api-m.sandbox.paypal.com",
            },
        )
        self.env_patch.start()
        pp.init_database()
        self.token = patch.object(pp, "get_access_token", return_value="unit-test-token")
        self.token.start()

    def tearDown(self) -> None:
        self.token.stop()
        self.env_patch.stop()
        pp.DATABASE_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def _captured_order(self, order_id="REFORDER1", days_ago=1.0, status="COMPLETED", capture_id="CAP1"):
        with pp._db() as c:
            c.execute(
                """INSERT INTO paypal_orders
                   (order_id, sku, description, amount, currency, status, approval_url, created_at,
                    capture_id, payer_email, captured_at, captured_amount, capture_currency)
                   VALUES (?, 'starter_demo', 'MerchantPulse Starter Demo', '5.00', 'USD', ?, 'https://x',
                           ?, ?, 'buyer@example.test', ?, '5.00', 'USD')""",
                (order_id, status, _iso(days_ago), capture_id if status == "COMPLETED" else None, _iso(days_ago)),
            )

    def test_refund_success_records_audit_row_and_updates_ledger(self):
        self._captured_order()
        ok = _resp(201, {"id": "RF1", "status": "COMPLETED"})
        with patch.object(pp.requests, "post", return_value=ok) as post:
            result = pp.refund_capture("REFORDER1", reason="Charged twice", proposed_by="ai", ai_summary="AI: refund")
        url = post.call_args.args[0]
        self.assertTrue(url.endswith("/v2/payments/captures/CAP1/refund"))
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["PayPal-Request-Id"], "refund-REFORDER1")
        self.assertEqual(post.call_args.kwargs["json"]["amount"], {"value": "5.00", "currency_code": "USD"})
        self.assertEqual(result["status"], "COMPLETED")
        self.assertFalse(result["already_refunded"])
        row = pp.get_refund("REFORDER1")
        self.assertEqual(row["proposed_by"], "ai")
        self.assertEqual(row["refund_id"], "RF1")
        listed = pp.list_completed_payments()
        self.assertEqual(listed[0]["refund_status"], "COMPLETED")

    def test_second_refund_request_does_not_call_paypal(self):
        self._captured_order()
        with patch.object(pp.requests, "post", return_value=_resp(201, {"id": "RF1", "status": "COMPLETED"})):
            pp.refund_capture("REFORDER1", reason="Charged twice")
        with patch.object(pp.requests, "post") as post:
            again = pp.refund_capture("REFORDER1", reason="Charged twice")
        post.assert_not_called()
        self.assertTrue(again["already_refunded"])

    def test_unknown_order_is_rejected_before_network(self):
        with patch.object(pp.requests, "post") as post:
            with self.assertRaisesRegex(pp.PayPalIntegrationError, "not created by this MerchantPulse app"):
                pp.refund_capture("NOTOURS123", reason="x")
        post.assert_not_called()

    def test_invalid_order_id_and_blank_reason_are_rejected(self):
        self._captured_order()
        with self.assertRaises(pp.PayPalIntegrationError):
            pp.refund_capture("../../etc/passwd", reason="x")
        with self.assertRaisesRegex(pp.PayPalIntegrationError, "reason is required"):
            pp.refund_capture("REFORDER1", reason="   ")

    def test_uncaptured_order_cannot_be_refunded(self):
        self._captured_order(order_id="NOCAPTURE1", status="PAYER_ACTION_REQUIRED")
        with patch.object(pp.requests, "post") as post:
            with self.assertRaisesRegex(pp.PayPalIntegrationError, "blocked by policy"):
                pp.refund_capture("NOCAPTURE1", reason="x")
        post.assert_not_called()

    def test_outside_refund_window_is_blocked_by_policy(self):
        self._captured_order(order_id="OLDORDER1", days_ago=45)
        with patch.object(pp.requests, "post") as post:
            with self.assertRaisesRegex(pp.PayPalIntegrationError, "refund window"):
                pp.refund_capture("OLDORDER1", reason="Late request")
        post.assert_not_called()

    def test_timeout_then_paypal_confirms_refunded_is_recorded(self):
        self._captured_order()
        refunded = {
            "id": "REFORDER1", "status": "COMPLETED",
            "purchase_units": [{"payments": {
                "captures": [{"id": "CAP1", "status": "REFUNDED"}],
                "refunds": [{"id": "RF9", "status": "COMPLETED"}],
            }}],
        }
        with patch.object(pp.requests, "post", side_effect=pp.requests.Timeout("slow")), \
             patch.object(pp.requests, "get", return_value=_resp(200, refunded)):
            result = pp.refund_capture("REFORDER1", reason="Item never arrived")
        self.assertEqual(result["refund_id"], "RF9")
        self.assertEqual(pp.get_refund("REFORDER1")["status"], "COMPLETED")

    def test_timeout_without_confirmation_records_nothing(self):
        self._captured_order()
        not_refunded = {"id": "REFORDER1", "status": "COMPLETED",
                        "purchase_units": [{"payments": {"captures": [{"id": "CAP1", "status": "COMPLETED"}]}}]}
        with patch.object(pp.requests, "post", side_effect=pp.requests.Timeout("slow")), \
             patch.object(pp.requests, "get", return_value=_resp(200, not_refunded)):
            with self.assertRaisesRegex(pp.PayPalIntegrationError, "cannot refund twice"):
                pp.refund_capture("REFORDER1", reason="Item never arrived")
        self.assertIsNone(pp.get_refund("REFORDER1"))

    def test_paypal_says_fully_refunded_and_lookup_confirms(self):
        self._captured_order()
        refunded = {
            "id": "REFORDER1", "status": "COMPLETED",
            "purchase_units": [{"payments": {"captures": [{"id": "CAP1", "status": "REFUNDED"}],
                                             "refunds": [{"id": "RF7"}]}}],
        }
        err = _resp(422, {"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "CAPTURE_FULLY_REFUNDED"}]})
        with patch.object(pp.requests, "post", return_value=err), \
             patch.object(pp.requests, "get", return_value=_resp(200, refunded)):
            result = pp.refund_capture("REFORDER1", reason="Charged twice")
        self.assertEqual(result["refund_id"], "RF7")

    def test_paypal_error_without_refund_state_raises_and_records_nothing(self):
        self._captured_order()
        err = _resp(422, {"name": "UNPROCESSABLE_ENTITY", "message": "nope", "debug_id": "abc"})
        with patch.object(pp.requests, "post", return_value=err), \
             patch.object(pp.requests, "get", return_value=_resp(404, {})):
            with self.assertRaisesRegex(pp.PayPalIntegrationError, "refund failed"):
                pp.refund_capture("REFORDER1", reason="Charged twice")
        self.assertIsNone(pp.get_refund("REFORDER1"))

    def test_concurrent_refund_clicks_send_one_paypal_request(self):
        self._captured_order()
        started, release, calls = threading.Event(), threading.Event(), []

        def slow_post(*args, **kwargs):
            calls.append(kwargs)
            started.set()
            if not release.wait(timeout=5):
                raise AssertionError("test did not release the mocked refund request")
            return _resp(201, {"id": "RF1", "status": "COMPLETED"})

        with patch.object(pp.requests, "post", side_effect=slow_post):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(pp.refund_capture, "REFORDER1", reason="Charged twice")
                self.assertTrue(started.wait(timeout=3))
                second = pool.submit(pp.refund_capture, "REFORDER1", reason="Charged twice")
                time.sleep(0.15)
                release.set()
                results = [first.result(timeout=8), second.result(timeout=8)]
        self.assertEqual(len(calls), 1)
        self.assertEqual({r["already_refunded"] for r in results}, {False, True})

    def test_non_sandbox_url_is_refused(self):
        self._captured_order()
        with patch.dict("os.environ", {"PAYPAL_BASE_URL": "https://api-m.paypal.com"}):
            with patch.object(pp.requests, "post") as post:
                with self.assertRaisesRegex(pp.PayPalIntegrationError, "Sandbox only"):
                    pp.refund_capture("REFORDER1", reason="x")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
