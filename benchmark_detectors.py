"""Reproducible 100-dataset benchmark for the deterministic incident detectors.

Run with ``python benchmark_detectors.py``. This is a controlled synthetic test,
not evidence of real-world merchant performance. The 100 fixed cases contain 20
each of normal activity, card failure spikes, refund bursts, overnight high-value
orders, and a second no-incident cohort with a smaller transaction volume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from demo_data import make_demo_transactions
from detectors import run_all_detectors

TODAY = pd.Timestamp("2026-10-31")
COHORTS = ("normal", "failure_spike", "refund_burst", "offhours_high_value", "no_incident_small")
EXPECTED = {
    "failure_spike": "failure_spike",
    "refund_burst": "refund_burst",
    "offhours_high_value": "offhours_high_value",
}


def _case(seed: int, cohort: str) -> pd.DataFrame:
    rows, days = (300, 60) if cohort == "no_incident_small" else (650, 90)
    data = make_demo_transactions(days=days, rows=rows, today=TODAY, seed=seed, plant_scenarios=False)
    if cohort == "failure_spike":
        day = TODAY - pd.Timedelta(days=35)
        injected = pd.DataFrame({
            "Date": day + pd.to_timedelta(range(30), unit="h") / 5,
            "Amount (USD)": [float(35 + (i * 17) % 110) for i in range(30)],
            "Status": ["Failed"] * 24 + ["Completed"] * 6,
            "Category": ["Electronics"] * 30,
            "Payment Method": ["Card"] * 30,
        })
        data = pd.concat([data, injected], ignore_index=True)
    elif cohort == "refund_burst":
        dates, amounts = [], []
        for offset in (24, 23, 22):
            dates.extend([TODAY - pd.Timedelta(days=offset) + pd.Timedelta(hours=12, minutes=i) for i in range(10)])
            amounts.extend([float(90 + (i * 13) % 130) for i in range(10)])
        injected = pd.DataFrame({
            "Date": dates,
            "Amount (USD)": amounts,
            "Status": ["Refunded"] * len(dates),
            "Category": ["Electronics"] * len(dates),
            "Payment Method": ["Card"] * len(dates),
        })
        data = pd.concat([data, injected], ignore_index=True)
    elif cohort == "offhours_high_value":
        injected = pd.DataFrame({
            "Date": [TODAY - pd.Timedelta(days=18) + pd.Timedelta(hours=2), TODAY - pd.Timedelta(days=11) + pd.Timedelta(hours=3), TODAY - pd.Timedelta(days=5) + pd.Timedelta(hours=4)],
            "Amount (USD)": [3200.0, 3500.0, 3800.0],
            "Status": ["Completed"] * 3,
            "Category": ["Electronics", "Services", "Electronics"],
            "Payment Method": ["Card", "PayPal", "Card"],
        })
        data = pd.concat([data, injected], ignore_index=True)
    data = data.reset_index(drop=True)
    data["Transaction ID"] = [f"B{seed:03d}-{i:04d}" for i in range(len(data))]
    return data


def run_benchmark() -> tuple[dict[str, Any], pd.DataFrame]:
    cases = []
    findings_by_case: dict[str, list[dict[str, Any]]] = {}
    for cohort_index, cohort in enumerate(COHORTS):
        for seed_offset in range(20):
            seed = (seed_offset if cohort == "normal" else 20 + seed_offset
                    if cohort == "no_incident_small" else 1200 + (cohort_index - 1) * 20 + seed_offset)
            case_id = f"{cohort}-{seed}"
            findings = run_all_detectors(_case(seed, cohort))
            findings_by_case[case_id] = findings
            types = sorted({f["type"] for f in findings})
            attributable = [f for f in findings if f["type"] == EXPECTED.get(cohort)] if cohort in EXPECTED else findings
            cases.append({"case_id": case_id, "cohort": cohort, "seed": seed,
                          "expected_type": EXPECTED.get(cohort), "finding_count": len(findings),
                          "finding_types": ",".join(types),
                          "additional_finding_types": ",".join(sorted(set(types) - {EXPECTED.get(cohort)})) if cohort in EXPECTED else "",
                          "detected_expected": int(EXPECTED.get(cohort) in types) if cohort in EXPECTED else None,
                          "excess_failed_value_usd": round(sum(f.get("excess_failed_value_usd", 0) for f in attributable), 2),
                          "excess_refund_value_usd": round(sum(f.get("excess_refund_value_usd", 0) for f in attributable), 2),
                          "value_under_review_usd": round(sum(f.get("flagged_value_usd", 0) for f in attributable), 2)})
    results = pd.DataFrame(cases)
    no_incident = results[results.cohort.isin(["normal", "no_incident_small"])]
    positives = results[results.expected_type.notna()]
    fp_cases = no_incident[no_incident.finding_count > 0]
    missed = positives[positives.detected_expected == 0]
    baseline_sweep = []
    for seed in range(100):
        findings = run_all_detectors(make_demo_transactions(
            days=90, rows=650, today=TODAY, seed=seed, plant_scenarios=False
        ))
        if findings:
            baseline_sweep.append({
                "seed": seed,
                "alerts": [{"type": f["type"], "title": f["title"], "z_score": f.get("z_score"),
                            "summary": f["summary"],
                            "excess_failed_value_usd": f.get("excess_failed_value_usd", 0),
                            "excess_refund_value_usd": f.get("excess_refund_value_usd", 0)} for f in findings],
            })
    metrics = {
        "benchmark": "MerchantPulse controlled synthetic incident benchmark",
        "date": TODAY.date().isoformat(),
        "seed_rule": "normal seeds 0-19; small no-incident seeds 20-39; planted seeds 1200-1259",
        "dataset_count": len(results),
        "cohort_counts": {name: int((results.cohort == name).sum()) for name in COHORTS},
        "incident_detection_rate": round(float(positives.detected_expected.mean()), 4),
        "incident_cases": int(len(positives)),
        "detected_incident_cases": int(positives.detected_expected.sum()),
        "false_positive_dataset_count": int(len(fp_cases)),
        "no_incident_dataset_count": int(len(no_incident)),
        "false_positive_dataset_rate": round(float(len(fp_cases) / len(no_incident)), 4),
        "false_positive_findings": int(no_incident.finding_count.sum()),
        "missed_incidents": [{"case_id": r.case_id, "expected_type": r.expected_type} for r in missed.itertuples()],
        "false_positive_cases": [
            {"case_id": r.case_id, "finding_types": r.finding_types,
             "finding_titles": [f["title"] for f in findings_by_case[r.case_id]]}
            for r in fp_cases.itertuples()
        ],
        "no_incident_100_seed_investigation": {
            "description": "Supplemental same-size 90-day/650-transaction baseline sweep over seeds 0-99, with no planted incidents.",
            "case_count": 100,
            "false_positive_case_count": len(baseline_sweep),
            "false_positive_dataset_rate": round(len(baseline_sweep) / 100, 4),
            "cases": baseline_sweep,
            "interpretation": "These alerts are false positives relative to the synthetic no-incident labels. Their patterns are random clusters that cross the configured thresholds; this does not establish whether similar patterns would be actionable in real merchant data.",
        },
        "financial_value_associated_with_detected_incidents_usd": {
            "excess_failed_payment_value": round(float(positives.excess_failed_value_usd.sum()), 2),
            "excess_refund_value": round(float(positives.excess_refund_value_usd.sum()), 2),
            "transactions_under_review": round(float(positives.value_under_review_usd.sum()), 2),
        },
        "limitations": [
            "All 100 datasets are synthetic and generated from the project demo-data generator.",
            "Injected patterns are controlled examples, not representative merchant behavior.",
            "Financial values are detector-associated amounts, not realised loss, recovered revenue, or causal estimates.",
            "A case-level hit requires at least one finding of the expected injected type; extra alert types are counted and shown.",
        ],
    }
    return metrics, results


def main() -> None:
    metrics, cases = run_benchmark()
    out = Path(__file__).resolve().parent / "benchmark_results"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cases.to_csv(out / "cases.csv", index=False)
    lines = [
        "# MerchantPulse synthetic incident benchmark", "",
        f"Synthetic data anchor date: {metrics['date']}  ",
        f"Cases: {metrics['dataset_count']} (20 per cohort)  ",
        "This is controlled synthetic validation, not customer validation or a real-world accuracy claim.", "",
        "## Results", "",
        f"- Expected incident cases detected: {metrics['detected_incident_cases']}/{metrics['incident_cases']} ({metrics['incident_detection_rate']:.0%})",
        f"- No-incident datasets with one or more alerts: {metrics['false_positive_dataset_count']}/{metrics['no_incident_dataset_count']} ({metrics['false_positive_dataset_rate']:.0%})",
        f"- False-positive findings across those datasets: {metrics['false_positive_findings']}",
        f"- Missed planted incidents: {len(metrics['missed_incidents'])}",
        "", "## Detection by incident type", "",
        "| Incident type | Detected | Cases | Rate |", "| --- | ---: | ---: | ---: |",
    ]
    for kind in EXPECTED:
        part = cases[cases.cohort == kind]
        count = int(part.detected_expected.sum())
        lines.append(f"| `{kind}` | {count} | {len(part)} | {count / len(part):.0%} |")
    lines += ["", "## Financial amounts associated with detected incident cases", "",
              "These are amounts tagged by the detectors; they are not realised loss or recovered revenue.", "",
              "| Measure | Amount (USD) |", "| --- | ---: |"]
    for label, value in metrics["financial_value_associated_with_detected_incidents_usd"].items():
        lines.append(f"| {label.replace('_', ' ').capitalize()} | ${value:,.2f} |")
    lines += ["", "## False-positive cases", ""]
    if metrics["false_positive_cases"]:
        lines += ["| Case | Alerts | Finding titles |", "| --- | --- | --- |"]
        for case in metrics["false_positive_cases"]:
            lines.append(f"| `{case['case_id']}` | {case['finding_types']} | {'; '.join(case['finding_titles'])} |")
    else:
        lines.append("No alerts in the no-incident cohorts.")
    sweep = metrics["no_incident_100_seed_investigation"]
    lines += ["", "## Investigation of the earlier five-alert result", "",
              f"A reproducible sweep of 100 unplanted, 90-day datasets at 650 transactions each found {sweep['false_positive_case_count']}/100 cases with alerts.", "",
              "| Seed | Detector finding | z-score | Baseline comparison | Excess amount tagged (USD) |", "| ---: | --- | ---: | --- | ---: |"]
    for case in sweep["cases"]:
        for alert in case["alerts"]:
            measure = alert["excess_failed_value_usd"] or alert["excess_refund_value_usd"]
            lines.append(f"| {case['seed']} | {alert['title']} | {alert['z_score']:.1f} | {alert['summary']} | ${measure:,.2f} |")
    lines += ["", sweep["interpretation"],
              "The alerts come from random category refund clusters or payment-method failure clusters that cross the existing minimum-gap and z-score thresholds. The detector scans many categories, methods and dates in each dataset, so a per-window z threshold does not control the overall false-alert chance. This is the main issue to address before presenting the detector as highly accurate."]
    lines += ["", "## Interpretation limits", ""] + [f"- {item}" for item in metrics["limitations"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'summary.md'}")
    print(f"Detection: {metrics['detected_incident_cases']}/{metrics['incident_cases']}; "
          f"false-positive datasets: {metrics['false_positive_dataset_count']}/{metrics['no_incident_dataset_count']} "
          f"(primary benchmark), {sweep['false_positive_case_count']}/100 (baseline sweep); "
          f"missed incidents: {len(metrics['missed_incidents'])}")


if __name__ == "__main__":
    main()
