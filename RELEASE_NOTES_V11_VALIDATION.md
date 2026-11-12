# MerchantPulse AI V11 — Validation and Decision Receipts

## Added

- A fixed-seed benchmark covering 100 synthetic datasets: normal activity, three planted incident types and a smaller no-incident cohort.
- A supplemental 100-seed baseline sweep that reproduces and itemizes the five no-incident alerts from seeds 5, 15, 53, 56 and 73.
- Separate dashboard measurements for excess failed-payment value, excess refund value and transactions under review.
- Persistent Decision Receipts for refund triage, recording the customer report summary, AI reasoning, complaint safety-gate result, refund policy checks, approval state and PayPal execution result.
- Problem framing for small online merchants without dedicated payment-operations teams. Subscription-survey scope remains explicit.

## Benchmark summary

- Planted incident cases detected: 60/60.
- No-incident cases with alerts in the 100-case benchmark: 4/40.
- False-positive cases in the supplemental 100-seed baseline sweep: 5/100.
- Missed planted incident cases: 0.

All cases are synthetic. Financial values are amounts associated with detected patterns; they are not realised loss, recovered revenue or causal estimates. See `benchmark_results/summary.md` for the seed-level evidence and caveats.
