"""Offline unit tests for MerchantPulse PayPal integration helpers."""
from __future__ import annotations

import tempfile
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import paypal_integration as pp


class PayPalIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = pp.DATABASE_PATH
        pp.DATABASE_PATH = Path(self.temp_dir.name) / "test_payments.sqlite3"
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

    def tearDown(self) -> None:
        self.env_patch.stop()
        pp.DATABASE_PATH = self.original_db_path
        self.temp_dir.cleanup()

    @staticmethod
    def response(status_code: int, payload: dict) -> Mock:
        obj = Mock()
        obj.status_code = status_code
        obj.json.return_value = payload
        return obj

    def test_credentials_are_detected_without_exposing_them(self) -> None:
        self.assertTrue(pp.credentials_configured())

    def test_create_order_uses_server_side_catalog_and_saves_order(self) -> None:
        token_response = self.response(200, {"access_token": "unit-test-token"})
        order_response = self.response(
            201,
            {
                "id": "TESTORDER123",
                "status": "PAYER_ACTION_REQUIRED",
                "links": [{"rel": "payer-action", "href": "https://www.sandbox.paypal.com/checkoutnow?token=TESTORDER123"}],
            },
        )
        with patch.object(pp.requests, "post", side_effect=[token_response, order_response]) as post:
            result = pp.create_order(
                "starter_demo",
                return_url="http://localhost:8501/?paypal_return=1",
                cancel_url="http://localhost:8501/?paypal_cancel=1",
            )
        self.assertEqual(result["amount"], "5.00")
        self.assertEqual(result["order_id"], "TESTORDER123")
        sent_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(sent_payload["purchase_units"][0]["amount"]["value"], "5.00")
        self.assertNotIn("unit-test-client-secret", str(sent_payload))
        saved = pp.get_order("TESTORDER123")
        self.assertIsNotNone(saved)
        self.assertEqual(saved["status"], "PAYER_ACTION_REQUIRED")

    def test_create_order_rejects_unknown_product_before_network(self) -> None:
        with patch.object(pp.requests, "post") as post:
            with self.assertRaises(pp.PayPalIntegrationError):
                pp.create_order(
                    "arbitrary_price",
                    return_url="http://localhost:8501/",
                    cancel_url="http://localhost:8501/",
                )
        post.assert_not_called()

    def test_capture_rejects_order_not_created_by_this_app(self) -> None:
        with self.assertRaisesRegex(pp.PayPalIntegrationError, "not created by this MerchantPulse app"):
            pp.capture_order("NOTOURORDER123")

    def test_capture_saves_completed_payment(self) -> None:
        with pp._db() as connection:
            connection.execute(
                """INSERT INTO paypal_orders
                   (order_id, sku, description, amount, currency, status, approval_url, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("TESTORDER456", "starter_demo", "MerchantPulse Starter Demo", "5.00", "USD",
                 "PAYER_ACTION_REQUIRED", "https://sandbox.example/approve", "2026-10-30T00:00:00+00:00"),
            )
        token_response = self.response(200, {"access_token": "unit-test-token"})
        capture_response = self.response(
            201,
            {
                "id": "TESTORDER456",
                "status": "COMPLETED",
                "payer": {"email_address": "sandbox-buyer@example.test"},
                "purchase_units": [{"payments": {"captures": [{
                    "id": "TESTCAPTURE456", "status": "COMPLETED",
                    "amount": {"currency_code": "USD", "value": "5.00"},
                    "create_time": "2026-10-30T10:00:00Z",
                }]}}],
            },
        )
        with patch.object(pp.requests, "post", side_effect=[token_response, capture_response]):
            result = pp.capture_order("TESTORDER456")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["capture_id"], "TESTCAPTURE456")
        self.assertEqual(len(pp.list_completed_payments()), 1)
        # A second call returns the local result and does not call PayPal again.
        with patch.object(pp.requests, "post") as post:
            repeated = pp.capture_order("TESTORDER456")
        post.assert_not_called()
        self.assertTrue(repeated["already_captured"])


    def _insert_pending_order(self, order_id: str) -> None:
        with pp._db() as connection:
            connection.execute(
                """INSERT INTO paypal_orders
                   (order_id, sku, description, amount, currency, status, approval_url, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (order_id, "starter_demo", "MerchantPulse Starter Demo", "5.00", "USD",
                 "PAYER_ACTION_REQUIRED", "https://sandbox.example/approve", "2026-10-30T00:00:00+00:00"),
            )

    @staticmethod
    def _completed_order_payload(order_id: str) -> dict:
        return {
            "id": order_id,
            "status": "COMPLETED",
            "payer": {"email_address": "sandbox-buyer@example.test"},
            "purchase_units": [{"payments": {"captures": [{
                "id": f"CAPTURE-{order_id}",
                "status": "COMPLETED",
                "amount": {"currency_code": "USD", "value": "5.00"},
                "create_time": "2026-10-30T10:00:00Z",
            }]}}],
        }

    def test_capture_recovers_after_timeout_when_paypal_confirms_completed(self) -> None:
        self._insert_pending_order("TIMEOUTORDER1")
        with patch.object(pp, "get_access_token", return_value="unit-test-token"), \
             patch.object(pp.requests, "post", side_effect=pp.requests.Timeout("read timed out")), \
             patch.object(pp.requests, "get", return_value=self.response(
                 200, self._completed_order_payload("TIMEOUTORDER1")
             )):
            result = pp.capture_order("TIMEOUTORDER1")

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["capture_id"], "CAPTURE-TIMEOUTORDER1")
        self.assertEqual(len(pp.list_completed_payments()), 1)

    def test_capture_timeout_without_completed_lookup_does_not_write_ledger(self) -> None:
        self._insert_pending_order("TIMEOUTORDER2")
        with patch.object(pp, "get_access_token", return_value="unit-test-token"), \
             patch.object(pp.requests, "post", side_effect=pp.requests.Timeout("read timed out")), \
             patch.object(pp.requests, "get", return_value=self.response(
                 200, {"id": "TIMEOUTORDER2", "status": "APPROVED"}
             )):
            with self.assertRaisesRegex(pp.PayPalIntegrationError, "could not be verified as completed"):
                pp.capture_order("TIMEOUTORDER2")

        saved = pp.get_order("TIMEOUTORDER2")
        self.assertEqual(saved["status"], "PAYER_ACTION_REQUIRED")
        self.assertIsNone(saved["capture_id"])
        self.assertEqual(pp.list_completed_payments(), [])

    def test_concurrent_capture_requests_only_send_one_paypal_capture(self) -> None:
        self._insert_pending_order("CONCURRENTORDER1")
        post_started = threading.Event()
        release_post = threading.Event()
        post_calls = []
        completed_payload = self._completed_order_payload("CONCURRENTORDER1")

        def fake_capture_post(*args, **kwargs):
            post_calls.append(kwargs)
            post_started.set()
            if not release_post.wait(timeout=5):
                raise AssertionError("Test did not release the mocked PayPal capture request")
            return self.response(201, completed_payload)

        with patch.object(pp, "get_access_token", return_value="unit-test-token"), \
             patch.object(pp.requests, "post", side_effect=fake_capture_post):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(pp.capture_order, "CONCURRENTORDER1")
                self.assertTrue(post_started.wait(timeout=3), "First capture did not reach PayPal")
                second = pool.submit(pp.capture_order, "CONCURRENTORDER1")
                time.sleep(0.15)
                release_post.set()
                first_result = first.result(timeout=8)
                second_result = second.result(timeout=8)

        self.assertEqual(len(post_calls), 1)
        self.assertEqual({first_result["already_captured"], second_result["already_captured"]}, {False, True})
        self.assertEqual(len(pp.list_completed_payments()), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
