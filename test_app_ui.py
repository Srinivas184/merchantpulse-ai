"""Offline UI tests: the app boots, tells its story on the first screen, and warns about the PIN only when public."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paypal_integration as pp
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent / "app.py")
PIN_WARNING = "Approval PIN is not configured"


class AppUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db = pp.DATABASE_PATH
        pp.DATABASE_PATH = Path(self.tmp.name) / "ui.sqlite3"
        self.env = patch.dict(os.environ, {
            "GEMINI_API_KEY": "", "MERCHANT_APPROVAL_PIN": "", "APP_BASE_URL": "http://localhost:8501",
            "PAYPAL_CLIENT_ID": "", "PAYPAL_CLIENT_SECRET": "",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        pp.DATABASE_PATH = self.original_db
        self.tmp.cleanup()

    def run_app(self) -> AppTest:
        return AppTest.from_file(APP, default_timeout=120).run()

    def test_first_screen_tells_the_detect_explain_triage_approve_story(self):
        at = self.run_app()
        self.assertEqual([e.value for e in at.exception], [])
        text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        for step in ("1 · Detect", "2 · Explain", "3 · Triage", "4 · Approve", "Today's briefing"):
            self.assertIn(step, text)
        self.assertIn("incident(s) found by code", text)

    def test_impact_panel_is_sourced_and_honest(self):
        at = self.run_app()
        self.assertIn("Who this is for and why it matters", [e.label for e in at.expander])
        body = " ".join(m.value for m in at.markdown)
        self.assertIn("743 billion", body)
        self.assertIn("not realised loss", body)
        self.assertIn("subscription", body)  # the survey's scope is stated

    def test_pin_warning_is_hidden_locally_and_shown_on_a_public_url_without_a_pin(self):
        local = self.run_app()
        self.assertFalse(any(PIN_WARNING in w.value for w in local.warning))
        with patch.dict(os.environ, {"APP_BASE_URL": "https://demo.example.com"}):
            public = self.run_app()
        self.assertTrue(any(PIN_WARNING in w.value for w in public.warning))
        with patch.dict(os.environ, {"APP_BASE_URL": "https://demo.example.com", "MERCHANT_APPROVAL_PIN": "long-unique-pin"}):
            protected = self.run_app()
        self.assertFalse(any(PIN_WARNING in w.value for w in protected.warning))


if __name__ == "__main__":
    unittest.main(verbosity=2)
