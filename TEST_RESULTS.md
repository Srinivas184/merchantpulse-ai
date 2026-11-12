# MerchantPulse AI V11 - Verification

Run on Python 3.13 with the pinned requirements (`python -W error::DeprecationWarning -m unittest discover`):

- **78 tests passed**, including the 3 Streamlit AppTest UI checks and 5 new Decision Receipt tests.
- End-to-end UI flow with PayPal mocked: triage, approve, exactly 1 PayPal refund call, audit row written, ledger shows the refund, Decision Receipt moves Pending -> PayPal confirmed; wrong PIN makes 0 PayPal calls.
- The detector benchmark (`python benchmark_detectors.py`) was re-run from scratch and reproduces the shipped `benchmark_results/` exactly (60/60 planted cases detected; 4/40 no-incident cases alerted; 5/100 in the baseline sweep).
- Not verified: live PayPal Sandbox refund and live Gemini calls (no keys in the verification environment). Run one live checkout + refund yourself before recording.

## Independent recheck after applying the supplied layout adjustment

- Python compilation (`python -m compileall -q .`): passed.
- 75 non-UI tests across seven modules: passed.
- Detector benchmark reproduced: 60/60 planted incident cases detected; 4/40 primary no-incident cases alerted; 5/100 supplemental baseline seeds alerted.
- The 3 Streamlit AppTest UI checks were not independently rerun in this environment because Streamlit was unavailable and package-index access failed. Run `python -m unittest -v test_app_ui` locally after installing the pinned requirements.
- No live PayPal Sandbox or live Gemini calls were made during this review.
