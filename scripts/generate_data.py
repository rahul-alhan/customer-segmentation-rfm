"""Generate synthetic transactions + feedback for segmentation demo."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POSITIVE_TEMPLATES = [
    "Great service, very happy with the product.",
    "Quick delivery and excellent quality.",
    "Will buy again, support was helpful.",
    "Love the new updates, easy to use.",
]
NEGATIVE_TEMPLATES = [
    "Slow delivery, considering a refund.",
    "Product broke after a week, want to cancel.",
    "Support never responded, frustrated.",
    "Too expensive for what you get.",
]
NEUTRAL_TEMPLATES = [
    "It's okay, does what it says.",
    "Average experience overall.",
    "Nothing special either way.",
]


def generate(n_customers: int, end_date: str = "2026-05-01", seed: int = 11):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date)

    # customer "type" influences both transactions and feedback
    types = rng.choice(["loyal", "casual", "at_risk", "lost"], n_customers, p=[0.20, 0.50, 0.20, 0.10])
    cust_ids = [f"c_{i:05d}" for i in range(n_customers)]

    txn_rows = []
    fb_rows = []
    for cid, t in zip(cust_ids, types):
        if t == "loyal":
            n_tx, mean_amt, recent_window = rng.integers(8, 25), rng.uniform(60, 140), 30
            sentiment_pool = POSITIVE_TEMPLATES * 4 + NEUTRAL_TEMPLATES
        elif t == "casual":
            n_tx, mean_amt, recent_window = rng.integers(2, 8), rng.uniform(30, 80), 90
            sentiment_pool = POSITIVE_TEMPLATES + NEUTRAL_TEMPLATES * 2 + NEGATIVE_TEMPLATES
        elif t == "at_risk":
            n_tx, mean_amt, recent_window = rng.integers(1, 5), rng.uniform(40, 100), 180
            sentiment_pool = NEGATIVE_TEMPLATES * 2 + NEUTRAL_TEMPLATES
        else:  # lost
            n_tx, mean_amt, recent_window = rng.integers(0, 2), rng.uniform(20, 60), 365
            sentiment_pool = NEGATIVE_TEMPLATES * 3

        for _ in range(n_tx):
            offset = rng.integers(0, max(1, recent_window))
            txn_rows.append({
                "customer_id": cid,
                "txn_date": end - pd.Timedelta(days=int(offset)),
                "amount": round(float(rng.normal(mean_amt, mean_amt * 0.15)), 2),
            })
        n_fb = rng.integers(0, 4)
        for _ in range(n_fb):
            fb_rows.append({
                "customer_id": cid,
                "feedback": str(rng.choice(sentiment_pool)),
                "created_at": end - pd.Timedelta(days=int(rng.integers(0, 365))),
            })

    return pd.DataFrame(txn_rows), pd.DataFrame(fb_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--customers", type=int, default=2000)
    p.add_argument("--out", default="data/")
    args = p.parse_args()

    txn, fb = generate(args.customers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    txn.to_parquet(out / "transactions.parquet", index=False)
    fb.to_parquet(out / "feedback.parquet", index=False)
    print(f"transactions={len(txn):,}  feedback={len(fb):,}  → {out}")


if __name__ == "__main__":
    main()
