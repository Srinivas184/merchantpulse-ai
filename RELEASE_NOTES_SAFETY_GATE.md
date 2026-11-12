# MerchantPulse AI V9 — Complaint Safety Gate release notes

## Changes

- Added a deterministic, local pre-AI complaint screen in `refund_agent.py`.
- Suspicious instruction-like complaints are routed to `needs_review` before Gemini or local keyword triage is called.
- Added Unicode NFKC normalisation and common zero-width character handling for the scan copy.
- The refund assistant UI labels safety-gated cases and does not show refund approval controls for them.
- Refund approval is now allowed only when the final proposal action is `refund_full`, the policy-check list is non-empty, and every check passes.
- The UI hides approval controls for `no_refund`, `needs_review`, and failed/missing policy checks.
- Added regression tests for blocked prompt-injection patterns, zero-width obfuscation, ordinary complaint false-positive checks, local-mode gating, no-AI-call behavior, and fail-closed refund eligibility.
- Updated README security documentation and added `docs/COMPLAINT_SAFETY_GATE.md`.

## Limitations

The gate is a conservative regex-based heuristic, not a complete prompt-injection defence. False positives and false negatives are possible. It does not scrub personal information from customer complaint text. Keep the downstream system prompt, response validation, refund policy enforcement, human approval, PIN, idempotency, and audit controls in place.

## Verification in this packaging environment

- Python compilation: passed for all project Python files.
- Non-UI unit tests: 70 passed.
- Three Streamlit `AppTest` tests were not run in this isolated packaging environment because Streamlit is not installed here. Run the full suite in the project environment after `pip install -r requirements.txt`.
- No live Gemini or PayPal requests were made during this code change.

## Test accounting

The project contains 73 test methods across seven test modules. This isolated environment ran 70 non-UI tests successfully. The three Streamlit `AppTest` checks were not run here because Streamlit was unavailable in the build environment; run the full suite on the target Windows environment after installing `requirements.txt`.
