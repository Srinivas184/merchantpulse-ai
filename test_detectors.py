"""Tests for planted demo scenarios, statistical detectors, security helpers and AI context."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

import detectors as dt
from demo_data import make_demo_transactions
from merchant_insights import build_merchant_context
from security import allow_event, clean_text, mask_email, verify_approval_pin, approval_pin_required

TODAY = pd.Timestamp("2026-11-09")


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_demo_transactions(today=TODAY)
        cls.findings = dt.run_all_detectors(cls.data)

    def test_demo_data_is_reproducible_with_unique_ids(self):
        again = make_demo_transactions(today=TODAY)
        pd.testing.assert_frame_equal(self.data, again)
        self.assertTrue(self.data["Transaction ID"].is_unique)

    def test_planted_incidents_are_found(self):
        types = {f["type"] for f in self.findings}
        self.assertEqual(types, {"failure_spike", "refund_burst", "offhours_high_value"})
        spike = next(f for f in self.findings if f["type"] == "failure_spike")
        self.assertEqual(spike["payment_method"], "Card")
        self.assertEqual(spike["date"], "2026-11-03")
        self.assertGreater(spike["failure_rate_pct"], 40)
        self.assertLess(spike["baseline_rate_pct"], 20)
        burst = next(f for f in self.findings if f["type"] == "refund_burst")
        self.assertEqual(burst["category"], "Electronics")
        self.assertEqual(burst["start_date"], "2026-11-05")
        self.assertEqual(burst["end_date"], "2026-11-07")

    def test_no_false_alarms_without_planted_scenarios(self):
        for seed in (1, 2, 3, 7, 42, 99):
            base = make_demo_transactions(today=TODAY, seed=seed, plant_scenarios=False)
            self.assertEqual(dt.run_all_detectors(base), [], f"seed {seed}")

    def test_findings_are_sorted_critical_first_and_carry_evidence(self):
        self.assertEqual(self.findings[0]["severity"], "critical")
        for f in self.findings:
            self.assertTrue(f["summary"])
            self.assertTrue(f["transaction_ids"])

    def test_impact_numbers_are_the_sum_of_finding_evidence(self):
        impact = dt.summarize_impact(self.findings)
        self.assertEqual(impact["incident_count"], 3)
        expected = sum(f.get("excess_failed_value_usd", 0) + f.get("excess_refund_value_usd", 0) for f in self.findings)
        self.assertAlmostEqual(impact["estimated_excess_loss_usd"], round(expected, 2), places=2)

    def test_date_range_filter(self):
        inside = dt.findings_in_range(self.findings, "2026-11-01", "2026-11-09")
        self.assertEqual(len(inside), 3)
        outside = dt.findings_in_range(self.findings, "2026-10-01", "2026-10-20")
        self.assertEqual(outside, [])

    def test_small_or_empty_inputs_do_not_crash(self):
        self.assertEqual(dt.run_all_detectors(self.data.iloc[0:0]), [])
        self.assertEqual(dt.run_all_detectors(self.data.head(5)), [])


class AIContextTests(unittest.TestCase):
    def test_findings_reach_the_ai_without_transaction_ids(self):
        data = make_demo_transactions(today=TODAY)
        findings = dt.run_all_detectors(data)
        context = build_merchant_context(data, [], findings)
        text = str(context)
        summary = context["analytics_dataset"]["detected_findings"]
        self.assertEqual(len(summary), 3)
        self.assertIn("failure_rate_pct", text)
        self.assertNotIn("MP-1", text)  # no transaction IDs
        self.assertNotIn("transaction_ids", text)
        self.assertNotIn("summary", text)  # free text stays local

    def test_sandbox_refunds_are_summarised_separately(self):
        data = make_demo_transactions(today=TODAY)
        payments = [
            {"description": "A", "amount": "5.00", "currency": "USD", "refund_status": "COMPLETED"},
            {"description": "B", "amount": "15.00", "currency": "USD", "refund_status": None},
        ]
        sandbox = build_merchant_context(data, payments)["paypal_sandbox_ledger"]
        self.assertEqual(sandbox["completed_capture_count"], 2)
        self.assertEqual(sandbox["refunded_capture_count"], 1)
        self.assertEqual(sandbox["refunded_volume_by_currency"], {"USD": 5.0})


class SecurityHelperTests(unittest.TestCase):
    def test_mask_email(self):
        self.assertEqual(mask_email("buyer@example.com"), "b***@example.com")
        self.assertEqual(mask_email(None), "—")
        self.assertEqual(mask_email("not-an-email"), "—")

    def test_pin_is_open_when_unset_and_enforced_when_set(self):
        with patch.dict(os.environ, {"MERCHANT_APPROVAL_PIN": ""}):
            self.assertFalse(approval_pin_required())
            self.assertTrue(verify_approval_pin("anything"))
        with patch.dict(os.environ, {"MERCHANT_APPROVAL_PIN": "4321"}):
            self.assertTrue(approval_pin_required())
            self.assertTrue(verify_approval_pin(" 4321 "))
            self.assertFalse(verify_approval_pin("1234"))
            self.assertFalse(verify_approval_pin(None))
            self.assertFalse(verify_approval_pin(""))

    def test_rate_limiter(self):
        now = [0.0]
        events: list[float] = []
        clock = lambda: now[0]
        results = [allow_event(events, max_events=3, window_seconds=60, clock=clock) for _ in range(5)]
        self.assertEqual(results, [True, True, True, False, False])
        now[0] = 61.0
        self.assertTrue(allow_event(events, max_events=3, window_seconds=60, clock=clock))

    def test_clean_text(self):
        self.assertEqual(clean_text("a\x00b\n\n c", 10), "ab c")
        self.assertEqual(clean_text("x" * 50, 5), "xxxxx")
        self.assertEqual(clean_text(None, 5), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
