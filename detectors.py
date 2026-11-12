"""Deterministic merchant-risk detectors.

All maths happens here, in code. Gemini only explains the findings; it never
computes them. The failure-spike and refund-burst detectors compare observed rates
against the merchant's own baseline using a one-sided proportion z-test. The
overnight high-value detector uses a 99th-percentile amount threshold (with a USD
500 minimum) and a 00:00-05:59 time window. Each detector reports evidence so a
human can review the finding; no flag is proof of fraud.
"""
from __future__ import annotations

from math import sqrt
from typing import Any

import pandas as pd

Z_CRITICAL = 8.0


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    data["Amount (USD)"] = pd.to_numeric(data["Amount (USD)"], errors="coerce").fillna(0.0)
    data["_status"] = data["Status"].astype(str).str.casefold()
    data["_day"] = data["Date"].dt.normalize()
    return data


def _z(p_obs: float, p_base: float, n: int) -> float:
    """One-sided z for an observed rate against a baseline rate."""
    p_base = min(max(p_base, 0.01), 0.99)
    return (p_obs - p_base) / sqrt(p_base * (1 - p_base) / n)


def _severity(z: float) -> str:
    return "critical" if z >= Z_CRITICAL else "warning"


def detect_failure_spikes(
    df: pd.DataFrame, *, z_threshold: float = 4.0, min_settled: int = 8, min_gap: float = 0.20
) -> list[dict[str, Any]]:
    """Days where one payment method's failure rate is far above its own baseline."""
    data = _prepare(df)
    settled = data[data["_status"].isin(["completed", "failed"])]
    findings: list[dict[str, Any]] = []
    for method, group in settled.groupby("Payment Method"):
        total_n = len(group)
        total_failed = int((group["_status"] == "failed").sum())
        for day, day_group in group.groupby("_day"):
            n = len(day_group)
            if n < min_settled:
                continue
            failed = day_group[day_group["_status"] == "failed"]
            f = len(failed)
            rest_n = total_n - n
            if rest_n < 20:
                continue
            baseline = (total_failed - f) / rest_n
            rate = f / n
            z = _z(rate, baseline, n)
            if z < z_threshold or rate - baseline < min_gap:
                continue
            avg_failed = float(failed["Amount (USD)"].mean()) if f else 0.0
            excess_failures = max(0.0, f - n * baseline)
            findings.append(
                {
                    "type": "failure_spike",
                    "severity": _severity(z),
                    "title": f"{method} failure spike on {day:%d %b %Y}",
                    "date": day.date().isoformat(),
                    "start_date": day.date().isoformat(),
                    "end_date": day.date().isoformat(),
                    "payment_method": str(method),
                    "settled_count": n,
                    "failed_count": f,
                    "failure_rate_pct": round(rate * 100, 1),
                    "baseline_rate_pct": round(baseline * 100, 1),
                    "z_score": round(z, 1),
                    "failed_value_usd": round(float(failed["Amount (USD)"].sum()), 2),
                    "excess_failed_value_usd": round(excess_failures * avg_failed, 2),
                    "transaction_ids": failed["Transaction ID"].astype(str).tolist()
                    if "Transaction ID" in failed.columns
                    else [],
                    "summary": (
                        f"{f} of {n} {method} payments failed ({rate * 100:.0f}% vs a "
                        f"{baseline * 100:.0f}% baseline)."
                    ),
                }
            )
    return findings


def detect_refund_bursts(
    df: pd.DataFrame,
    *,
    window_days: int = 3,
    z_threshold: float = 4.0,
    min_n: int = 10,
    min_gap: float = 0.20,
) -> list[dict[str, Any]]:
    """Rolling windows where a category's refund rate is far above its own baseline."""
    data = _prepare(df)
    findings: list[dict[str, Any]] = []
    for category, group in data.groupby("Category"):
        daily = group.groupby("_day").agg(
            n=("_status", "size"), r=("_status", lambda s: int((s == "refunded").sum()))
        )
        full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_range, fill_value=0)
        rolling = daily.rolling(window_days, min_periods=window_days).sum().dropna()
        total_n, total_r = int(daily["n"].sum()), int(daily["r"].sum())
        best: tuple[float, pd.Timestamp, int, int, float] | None = None
        for end, row in rolling.iterrows():
            n, r = int(row["n"]), int(row["r"])
            rest_n = total_n - n
            if n < min_n or rest_n < 20:
                continue
            baseline = (total_r - r) / rest_n
            rate = r / n
            z = _z(rate, baseline, n)
            if z >= z_threshold and rate - baseline >= min_gap and (best is None or z > best[0]):
                best = (z, end, n, r, baseline)
        if best is None:
            continue
        z, end, n, r, baseline = best
        start = end - pd.Timedelta(days=window_days - 1)
        window = group[(group["_day"] >= start) & (group["_day"] <= end) & (group["_status"] == "refunded")]
        avg_refund = float(window["Amount (USD)"].mean()) if len(window) else 0.0
        findings.append(
            {
                "type": "refund_burst",
                "severity": _severity(z),
                "title": f"{category} refund burst, {start:%d %b}–{end:%d %b %Y}",
                "date": start.date().isoformat(),
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "category": str(category),
                "total_count": n,
                "refunded_count": r,
                "refund_rate_pct": round(r / n * 100, 1),
                "baseline_rate_pct": round(baseline * 100, 1),
                "z_score": round(z, 1),
                "refund_value_usd": round(float(window["Amount (USD)"].sum()), 2),
                "excess_refund_value_usd": round(max(0.0, r - n * baseline) * avg_refund, 2),
                "transaction_ids": window["Transaction ID"].astype(str).tolist()
                if "Transaction ID" in window.columns
                else [],
                "summary": (
                    f"{r} of {n} {category} transactions were refunded "
                    f"({r / n * 100:.0f}% vs a {baseline * 100:.0f}% baseline)."
                ),
            }
        )
    return findings


def detect_offhours_high_value(
    df: pd.DataFrame, *, quantile: float = 0.99, floor_usd: float = 500.0, min_count: int = 2
) -> list[dict[str, Any]]:
    """Unusually large completed orders placed between 00:00 and 05:59."""
    data = _prepare(df)
    completed = data[data["_status"] == "completed"]
    if completed.empty:
        return []
    threshold = max(float(completed["Amount (USD)"].quantile(quantile)), floor_usd)
    flagged = completed[(completed["Amount (USD)"] >= threshold) & (completed["Date"].dt.hour < 6)]
    if len(flagged) < min_count:
        return []
    first, last = flagged["_day"].min(), flagged["_day"].max()
    return [
        {
            "type": "offhours_high_value",
            "severity": "warning",
            "title": "High-value orders placed overnight",
            "date": first.date().isoformat(),
            "start_date": first.date().isoformat(),
            "end_date": last.date().isoformat(),
            "flagged_count": int(len(flagged)),
            "flagged_value_usd": round(float(flagged["Amount (USD)"].sum()), 2),
            "amount_threshold_usd": round(threshold, 2),
            "hour_window": "00:00-05:59",
            "z_score": 0.0,
            "transaction_ids": flagged["Transaction ID"].astype(str).tolist()
            if "Transaction ID" in flagged.columns
            else [],
            "summary": (
                f"{len(flagged)} completed orders of at least ${threshold:,.0f} were placed "
                f"between 00:00 and 05:59 (total ${flagged['Amount (USD)'].sum():,.2f}). "
                "Review them; this is not proof of fraud."
            ),
        }
    ]


def run_all_detectors(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run every detector; critical findings first, then by strength of evidence."""
    findings = detect_failure_spikes(df) + detect_refund_bursts(df) + detect_offhours_high_value(df)
    return sorted(findings, key=lambda f: (f["severity"] != "critical", -float(f.get("z_score", 0.0))))


def findings_in_range(findings: list[dict[str, Any]], start: Any, end: Any) -> list[dict[str, Any]]:
    """Keep findings whose date span overlaps [start, end] (date-like values)."""
    s, e = pd.Timestamp(start).date().isoformat(), pd.Timestamp(end).date().isoformat()
    return [f for f in findings if f["end_date"] >= s and f["start_date"] <= e]


def summarize_impact(findings: list[dict[str, Any]]) -> dict[str, float]:
    """Headline numbers for the findings (all derived from the evidence above)."""
    excess_failed = sum(f.get("excess_failed_value_usd", 0.0) for f in findings)
    excess_refund = sum(f.get("excess_refund_value_usd", 0.0) for f in findings)
    under_review = sum(f.get("flagged_value_usd", 0.0) for f in findings)
    return {
        "incident_count": len(findings),
        "excess_failed_value_usd": round(excess_failed, 2),
        "excess_refund_value_usd": round(excess_refund, 2),
        # Kept for old callers; the UI reports the separate measures above because
        # their combined total can be mistaken for realised or recoverable loss.
        "estimated_excess_loss_usd": round(excess_failed + excess_refund, 2),
        "value_under_review_usd": round(under_review, 2),
    }
