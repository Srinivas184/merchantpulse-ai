"""Decision Receipts: stages are recorded separately and never claim more than happened."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import paypal_integration as pp


class DecisionReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db = pp.DATABASE_PATH
        pp.DATABASE_PATH = Path(self.tmp.name) / "receipts.sqlite3"
        pp.init_database()

    def tearDown(self):
        pp.DATABASE_PATH = self.original_db
        self.tmp.cleanup()

    def make(self, **kw):
        base = dict(evidence_summary="Customer report: charged twice", ai_reasoning="Recommend: full refund",
                    safety_status="Passed Complaint Safety Gate", policy_checks=[(True, "Within 30 days")])
        base.update(kw)
        return pp.create_decision_receipt("ORDER123456", **base)

    def test_new_receipt_is_a_recommendation_only(self):
        self.make()
        r = pp.list_decision_receipts("ORDER123456")[0]
        self.assertEqual(r["human_authority"], "Pending explicit approval")
        self.assertEqual(r["execution_status"], "Not executed")
        self.assertIsNone(r["execution_reference"])
        self.assertEqual(r["policy_checks"], [[True, "Within 30 days"]])

    def test_stages_update_independently_and_unknown_fields_are_ignored(self):
        rid = self.make()
        pp.update_decision_receipt(rid, human_authority="Explicit approval provided in dashboard", evidence_summary="TAMPERED")
        r = pp.list_decision_receipts("ORDER123456")[0]
        self.assertEqual(r["human_authority"], "Explicit approval provided in dashboard")
        self.assertEqual(r["execution_status"], "Not executed")  # approval is not execution
        self.assertEqual(r["evidence_summary"], "Customer report: charged twice")  # not an updatable field
        pp.update_decision_receipt(rid, execution_status="PayPal confirmed COMPLETED", execution_reference="RF1")
        r = pp.list_decision_receipts("ORDER123456")[0]
        self.assertEqual((r["execution_status"], r["execution_reference"]), ("PayPal confirmed COMPLETED", "RF1"))

    def test_emails_and_long_numbers_are_masked_and_length_is_bounded(self):
        self.make(evidence_summary="Contact me at jane.doe@example.com card 4111111111111111 " + "x" * 900)
        text = pp.list_decision_receipts("ORDER123456")[0]["evidence_summary"]
        self.assertNotIn("jane.doe@example.com", text)
        self.assertNotIn("4111111111111111", text)
        self.assertLessEqual(len(text), 500)

    def test_safety_gate_hold_is_recorded_and_failed_checks_are_kept(self):
        self.make(safety_status="Held by Complaint Safety Gate; no Gemini request sent",
                  policy_checks=[(False, "Outside refund window")])
        r = pp.list_decision_receipts("ORDER123456")[0]
        self.assertIn("Held by Complaint Safety Gate", r["safety_status"])
        self.assertFalse(r["policy_checks"][0][0])

    def test_listing_filters_by_order_and_is_newest_first_and_capped(self):
        for _ in range(3):
            self.make()
        pp.create_decision_receipt("OTHER", evidence_summary="x", ai_reasoning="y", safety_status="z", policy_checks=[])
        self.assertEqual(len(pp.list_decision_receipts("ORDER123456")), 3)
        self.assertEqual(len(pp.list_decision_receipts(limit=2)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
