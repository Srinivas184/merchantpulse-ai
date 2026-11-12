"""Offline tests for MerchantPulse local merchant insights."""
import unittest

import pandas as pd

from merchant_insights import answer_merchant_question


class MerchantInsightsTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            [
                {"Transaction ID": "MP-1", "Date": "2026-10-01", "Amount (USD)": 100.0, "Status": "Completed", "Category": "Electronics", "Payment Method": "Card", "ML Anomaly": False, "Anomaly Score": 0.1},
                {"Transaction ID": "MP-2", "Date": "2026-10-02", "Amount (USD)": 50.0, "Status": "Completed", "Category": "Clothing", "Payment Method": "PayPal", "ML Anomaly": True, "Anomaly Score": 0.8},
                {"Transaction ID": "MP-3", "Date": "2026-10-03", "Amount (USD)": 25.0, "Status": "Failed", "Category": "Electronics", "Payment Method": "Card", "ML Anomaly": False, "Anomaly Score": 0.2},
                {"Transaction ID": "MP-4", "Date": "2026-10-04", "Amount (USD)": 20.0, "Status": "Refunded", "Category": "Clothing", "Payment Method": "PayPal", "ML Anomaly": False, "Anomaly Score": 0.1},
                {"Transaction ID": "MP-5", "Date": "2026-10-05", "Amount (USD)": 40.0, "Status": "Pending", "Category": "Food", "Payment Method": "Pay Later", "ML Anomaly": False, "Anomaly Score": 0.1},
            ]
        )
        self.data["Date"] = pd.to_datetime(self.data["Date"])

    def test_category_question_uses_completed_revenue(self):
        answer = answer_merchant_question("Which category generates the most revenue?", self.data)
        self.assertIn("Electronics", answer)
        self.assertIn("$100.00", answer)

    def test_anomaly_question_names_flagged_transaction(self):
        answer = answer_merchant_question("Show unusual transactions", self.data)
        self.assertIn("MP-2", answer)
        self.assertIn("not fraud verdicts", answer)

    def test_failure_question_reports_rate(self):
        answer = answer_merchant_question("Why are payments failing?", self.data)
        self.assertIn("33.3%", answer)
        self.assertIn("Card", answer)

    def test_refund_question_reports_amount(self):
        answer = answer_merchant_question("Summarise refunds", self.data)
        self.assertIn("$20.00", answer)
        self.assertIn("Clothing", answer)

    def test_pending_question_reports_pending_amount(self):
        answer = answer_merchant_question("What is pending?", self.data)
        self.assertIn("1 pending", answer)
        self.assertIn("$40.00", answer)

    def test_business_improvement_question_returns_grounded_priorities(self):
        answer = answer_merchant_question(
            "How can I improve business performance based on these results?",
            self.data,
        )
        self.assertIn("evidence-based priorities", answer)
        self.assertIn("failed checkouts", answer)
        self.assertIn("refund concentration", answer)
        self.assertIn("leading revenue category", answer)
        self.assertIn("margins", answer)
        self.assertIn("not proof of fraud", answer)
        self.assertIn("synthetic analytics sample", answer)

    def test_empty_question_has_helpful_prompt(self):
        answer = answer_merchant_question("  ", self.data)
        self.assertIn("Enter a question", answer)

    def test_empty_dataset_is_handled(self):
        answer = answer_merchant_question("Show revenue", self.data.iloc[0:0])
        self.assertIn("no transactions", answer)


class MerchantAIGenerationTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            [
                {"Transaction ID": "MP-1", "Date": "2026-10-01", "Amount (USD)": 100.0, "Status": "Completed", "Category": "Electronics", "Payment Method": "Card", "ML Anomaly": False, "Anomaly Score": 0.1},
                {"Transaction ID": "MP-2", "Date": "2026-10-02", "Amount (USD)": 50.0, "Status": "Completed", "Category": "Clothing", "Payment Method": "PayPal", "ML Anomaly": True, "Anomaly Score": 0.8},
                {"Transaction ID": "MP-3", "Date": "2026-10-03", "Amount (USD)": 25.0, "Status": "Failed", "Category": "Electronics", "Payment Method": "Card", "ML Anomaly": False, "Anomaly Score": 0.2},
                {"Transaction ID": "MP-4", "Date": "2026-10-04", "Amount (USD)": 20.0, "Status": "Refunded", "Category": "Clothing", "Payment Method": "PayPal", "ML Anomaly": False, "Anomaly Score": 0.1},
                {"Transaction ID": "MP-5", "Date": "2026-10-05", "Amount (USD)": 40.0, "Status": "Pending", "Category": "Food", "Payment Method": "Pay Later", "ML Anomaly": False, "Anomaly Score": 0.1},
            ]
        )
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.paypal = [
            {
                "description": "MerchantPulse Starter Demo",
                "amount": "5.00",
                "currency": "USD",
                "status": "COMPLETED",
                "order_id": "ORDER-PRIVATE-1",
                "capture_id": "CAPTURE-PRIVATE-1",
                "payer_email": "buyer-private@example.test",
            },
            {
                "description": "MerchantPulse Growth Demo",
                "amount": "10.00",
                "currency": "USD",
                "status": "COMPLETED",
                "order_id": "ORDER-PRIVATE-2",
                "capture_id": "CAPTURE-PRIVATE-2",
                "payer_email": "buyer-private2@example.test",
            },
        ]

    def test_context_computes_analytics_and_sandbox_separately(self):
        from merchant_insights import build_merchant_context

        context = build_merchant_context(self.data, self.paypal)
        analytics = context["analytics_dataset"]
        sandbox = context["paypal_sandbox_ledger"]

        self.assertEqual(analytics["selected_transaction_count"], 5)
        self.assertEqual(analytics["completed_revenue_usd"], 150.0)
        self.assertEqual(analytics["status_counts"]["failed"], 1)
        self.assertEqual(analytics["failed_transactions_by_payment_method"][0]["count"], 1)
        self.assertEqual(sandbox["completed_capture_count"], 2)
        self.assertEqual(sandbox["captured_volume_by_currency"]["USD"], 15.0)
        self.assertTrue(sandbox["is_sandbox_test_data"])
        self.assertFalse(sandbox["real_money_moved"])
        self.assertIn("separate sources", context["source_separation_rule"])

    def test_context_excludes_paypal_buyer_and_transaction_identifiers(self):
        from merchant_insights import build_merchant_context

        context_text = str(build_merchant_context(self.data, self.paypal))
        for private_value in (
            "ORDER-PRIVATE-1",
            "CAPTURE-PRIVATE-1",
            "buyer-private@example.test",
            "ORDER-PRIVATE-2",
            "CAPTURE-PRIVATE-2",
        ):
            self.assertNotIn(private_value, context_text)
        self.assertIn("Buyer emails", context_text)
        # Synthetic transaction IDs and row-level anomaly examples are excluded too.
        self.assertNotIn("MP-2", context_text)
        self.assertNotIn("examples", context_text)


    def test_ai_uses_gemini_api_and_only_sends_aggregate_context(self):
        import json
        from types import SimpleNamespace
        from merchant_insights import answer_merchant_question_ai

        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Review the failure concentration and pending queue first."
                            )
                        )
                    ]
                )

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeClient:
            def __init__(self):
                self.chat = FakeChat()

        fake_client = FakeClient()
        answer = answer_merchant_question_ai(
            "How can I improve business performance?",
            self.data,
            self.paypal,
            client=fake_client,
            model="test-model",
        )

        self.assertIn("pending queue", answer)
        call = fake_client.chat.completions.kwargs
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["max_tokens"], 1600)
        self.assertNotIn("temperature", call)

        payload = json.loads(call["messages"][1]["content"])
        context = payload["dashboard_context"]

        self.assertEqual(context["analytics_dataset"]["completed_revenue_usd"], 150.0)
        self.assertEqual(
            context["paypal_sandbox_ledger"]["captured_volume_by_currency"]["USD"],
            15.0,
        )

        sent_content = call["messages"][1]["content"]
        self.assertNotIn("buyer-private@example.test", sent_content)
        self.assertNotIn("ORDER-PRIVATE-1", sent_content)
        self.assertNotIn("MP-2", sent_content)
        self.assertNotIn("locally_calculated_reference_answer_if_relevant", sent_content)


    def test_default_model_is_gemini_35_flash_lite_when_not_overridden(self):
        import os
        from unittest.mock import patch
        from types import SimpleNamespace
        from merchant_insights import answer_merchant_question_ai

        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="Summary based on aggregate dashboard facts.")
                        )
                    ]
                )

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeClient:
            def __init__(self):
                self.chat = FakeChat()

        fake_client = FakeClient()
        with patch.dict(os.environ, {"GEMINI_MODEL": ""}, clear=False):
            answer_merchant_question_ai(
                "Summarise the selected data",
                self.data,
                client=fake_client,
            )

        self.assertEqual(
            fake_client.chat.completions.kwargs["model"],
            "gemini-3.5-flash-lite",
        )


    def test_missing_api_key_raises_clear_error_without_network_call(self):
        import os
        from unittest.mock import patch
        from merchant_insights import MerchantAIError, answer_merchant_question_ai

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(
                MerchantAIError,
                "GEMINI_API_KEY is not configured",
            ):
                answer_merchant_question_ai("Summarise performance", self.data)


    def test_api_error_is_redacted(self):
        from merchant_insights import MerchantAIError, answer_merchant_question_ai

        class BrokenCompletions:
            def create(self, **kwargs):
                raise RuntimeError("sk-test-secret-must-not-leak")

        class BrokenChat:
            def __init__(self):
                self.completions = BrokenCompletions()

        class BrokenClient:
            def __init__(self):
                self.chat = BrokenChat()

        with self.assertRaises(MerchantAIError) as caught:
            answer_merchant_question_ai(
                "Summarise performance",
                self.data,
                client=BrokenClient(),
            )

        self.assertIn("RuntimeError", str(caught.exception))
        self.assertNotIn("sk-test-secret", str(caught.exception))

    def test_blank_question_uses_local_prompt_without_calling_ai(self):
        from merchant_insights import answer_merchant_question_ai

        class NeverCallResponses:
            def create(self, **kwargs):
                raise AssertionError("The AI API must not be called for a blank question")

        class NeverCallClient:
            responses = NeverCallResponses()

        answer = answer_merchant_question_ai("  ", self.data, client=NeverCallClient())
        self.assertIn("Enter a question", answer)


    def test_empty_model_output_is_rejected(self):
        from types import SimpleNamespace
        from merchant_insights import MerchantAIError, answer_merchant_question_ai

        class EmptyCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="   ")
                        )
                    ]
                )

        class EmptyChat:
            def __init__(self):
                self.completions = EmptyCompletions()

        class EmptyClient:
            def __init__(self):
                self.chat = EmptyChat()

        with self.assertRaisesRegex(MerchantAIError, "empty answer"):
            answer_merchant_question_ai(
                "Summarise performance",
                self.data,
                client=EmptyClient(),
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
