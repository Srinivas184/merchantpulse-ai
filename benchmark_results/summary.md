# MerchantPulse synthetic incident benchmark

Synthetic data anchor date: 2026-10-31  
Cases: 100 (20 per cohort)  
This is controlled synthetic validation, not customer validation or a real-world accuracy claim.

## Results

- Expected incident cases detected: 60/60 (100%)
- No-incident datasets with one or more alerts: 4/40 (10%)
- False-positive findings across those datasets: 4
- Missed planted incidents: 0

## Detection by incident type

| Incident type | Detected | Cases | Rate |
| --- | ---: | ---: | ---: |
| `failure_spike` | 20 | 20 | 100% |
| `refund_burst` | 20 | 20 | 100% |
| `offhours_high_value` | 20 | 20 | 100% |

## Financial amounts associated with detected incident cases

These are amounts tagged by the detectors; they are not realised loss or recovered revenue.

| Measure | Amount (USD) |
| --- | ---: |
| Excess failed payment value | $34,829.90 |
| Excess refund value | $83,647.65 |
| Transactions under review | $211,402.89 |

## False-positive cases

| Case | Alerts | Finding titles |
| --- | --- | --- |
| `normal-5` | refund_burst | Services refund burst, 20 Aug–22 Aug 2026 |
| `normal-15` | refund_burst | Clothing refund burst, 27 Sep–29 Sep 2026 |
| `no_incident_small-23` | refund_burst | Clothing refund burst, 17 Oct–19 Oct 2026 |
| `no_incident_small-29` | refund_burst | Clothing refund burst, 09 Oct–11 Oct 2026 |

## Investigation of the earlier five-alert result

A reproducible sweep of 100 unplanted, 90-day datasets at 650 transactions each found 5/100 cases with alerts.

| Seed | Detector finding | z-score | Baseline comparison | Excess amount tagged (USD) |
| ---: | --- | ---: | --- | ---: |
| 5 | Services refund burst, 20 Aug–22 Aug 2026 | 6.0 | 3 of 10 Services transactions were refunded (30% vs a 2% baseline). | $146.74 |
| 15 | Clothing refund burst, 27 Sep–29 Sep 2026 | 4.5 | 4 of 16 Clothing transactions were refunded (25% vs a 4% baseline). | $217.37 |
| 53 | PayPal failure spike on 01 Oct 2026 | 4.8 | 5 of 10 PayPal payments failed (50% vs a 8% baseline). | $268.85 |
| 56 | PayPal failure spike on 14 Oct 2026 | 4.7 | 5 of 9 PayPal payments failed (56% vs a 10% baseline). | $419.01 |
| 73 | Home & Living refund burst, 02 Sep–04 Sep 2026 | 4.6 | 3 of 10 Home & Living transactions were refunded (30% vs a 4% baseline). | $64.73 |

These alerts are false positives relative to the synthetic no-incident labels. Their patterns are random clusters that cross the configured thresholds; this does not establish whether similar patterns would be actionable in real merchant data.
The alerts come from random category refund clusters or payment-method failure clusters that cross the existing minimum-gap and z-score thresholds. The detector scans many categories, methods and dates in each dataset, so a per-window z threshold does not control the overall false-alert chance. This is the main issue to address before presenting the detector as highly accurate.

## Interpretation limits

- All 100 datasets are synthetic and generated from the project demo-data generator.
- Injected patterns are controlled examples, not representative merchant behavior.
- Financial values are detector-associated amounts, not realised loss, recovered revenue, or causal estimates.
- A case-level hit requires at least one finding of the expected injected type; extra alert types are counted and shown.
