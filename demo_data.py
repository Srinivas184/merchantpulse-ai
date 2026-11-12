"""Reproducible synthetic merchant transactions with planted, discoverable stories.

This is DEMO DATA, not PayPal data. On top of a random base it plants three
incidents relative to "today" so the detectors and the AI assistant have real
patterns to find (they are found by statistics, not hard-coded in the UI):

1. Card gateway incident: a burst of Card payments with a high failure rate.
2. Electronics refund burst: three days of unusually many refunds.
3. Off-hours high-value orders: large completed orders at 02:00-04:59.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["Electronics", "Clothing", "Home & Living", "Food", "Services"]
METHODS = ["PayPal", "Card", "Pay Later"]


def _amounts(rng: np.random.Generator, size: int) -> np.ndarray:
    return np.clip(rng.lognormal(mean=4.0, sigma=0.65, size=size).round(2), 5, 2500)


def make_demo_transactions(
    days: int = 90,
    rows: int = 650,
    *,
    today: pd.Timestamp | str | None = None,
    seed: int = 42,
    plant_scenarios: bool = True,
) -> pd.DataFrame:
    """Return synthetic transactions (newest first). Pass `today` for tests."""
    rng = np.random.default_rng(seed)
    end_day = (pd.Timestamp(today) if today is not None else pd.Timestamp.today()).normalize()
    start_day = end_day - pd.Timedelta(days=days - 1)

    timestamps = start_day + pd.to_timedelta(
        rng.integers(0, days, size=rows), unit="D"
    ) + pd.to_timedelta(rng.integers(0, 24 * 60, size=rows), unit="m")
    base = pd.DataFrame(
        {
            "Date": timestamps,
            "Amount (USD)": _amounts(rng, rows),
            "Status": rng.choice(
                ["Completed", "Pending", "Failed", "Refunded"], size=rows, p=[0.78, 0.08, 0.09, 0.05]
            ),
            "Category": rng.choice(CATEGORIES, size=rows, p=[0.22, 0.24, 0.18, 0.20, 0.16]),
            "Payment Method": rng.choice(METHODS, size=rows, p=[0.62, 0.28, 0.10]),
        }
    )
    parts = [base]

    if plant_scenarios:
        srng = np.random.default_rng(seed + 1)  # separate stream: base data never changes

        # 1) Card gateway incident, 6 days ago, 12:00-17:59, ~60% of Card payments fail.
        n = 30
        day = end_day - pd.Timedelta(days=6)
        parts.append(
            pd.DataFrame(
                {
                    "Date": day + pd.to_timedelta(srng.integers(12 * 60, 18 * 60, size=n), unit="m"),
                    "Amount (USD)": _amounts(srng, n),
                    "Status": srng.choice(["Failed", "Completed"], size=n, p=[0.6, 0.4]),
                    "Category": srng.choice(CATEGORIES, size=n),
                    "Payment Method": "Card",
                }
            )
        )

        # 2) Electronics refund burst over 4, 3 and 2 days ago.
        burst_days = [end_day - pd.Timedelta(days=d) for d in (4, 3, 2)]
        for d in burst_days:
            n = 8
            parts.append(
                pd.DataFrame(
                    {
                        "Date": d + pd.to_timedelta(srng.integers(9 * 60, 21 * 60, size=n), unit="m"),
                        "Amount (USD)": srng.uniform(80, 420, size=n).round(2),
                        "Status": "Refunded",
                        "Category": "Electronics",
                        "Payment Method": srng.choice(METHODS, size=n, p=[0.5, 0.35, 0.15]),
                    }
                )
            )

        # 3) Off-hours, unusually large completed orders ($900-$1,500 vs a ~$55 median).
        offhours = [(9, 2), (8, 3), (5, 2), (3, 4), (1, 3)]
        parts.append(
            pd.DataFrame(
                {
                    "Date": [
                        end_day - pd.Timedelta(days=d) + pd.Timedelta(hours=h, minutes=int(srng.integers(0, 60)))
                        for d, h in offhours
                    ],
                    "Amount (USD)": srng.uniform(900, 1500, size=len(offhours)).round(2),
                    "Status": "Completed",
                    "Category": ["Electronics", "Services", "Electronics", "Services", "Electronics"],
                    "Payment Method": ["Card", "Pay Later", "Card", "Card", "Pay Later"],
                }
            )
        )

    data = pd.concat(parts, ignore_index=True)
    data.insert(0, "Transaction ID", [f"MP-{100001 + i}" for i in range(len(data))])
    return data.sort_values("Date", ascending=False).reset_index(drop=True)
