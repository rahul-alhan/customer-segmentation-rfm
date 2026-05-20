"""Compute Recency, Frequency, Monetary scores per customer."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_rfm(transactions: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    transactions = transactions.copy()
    transactions["txn_date"] = pd.to_datetime(transactions["txn_date"])
    snap = pd.to_datetime(snapshot_date) if snapshot_date else transactions["txn_date"].max() + pd.Timedelta(days=1)

    g = transactions.groupby("customer_id")
    rfm = g.agg(
        recency_days=("txn_date", lambda s: (snap - s.max()).days),
        frequency=("txn_date", "count"),
        monetary=("amount", "sum"),
    ).reset_index()

    rfm["R"] = pd.qcut(rfm["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_string"] = "R" + rfm["R"].astype(str) + "_F" + rfm["F"].astype(str) + "_M" + rfm["M"].astype(str)
    return rfm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transactions", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    txn = pd.read_parquet(args.transactions)
    rfm = compute_rfm(txn)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(args.out, index=False)
    print(f"RFM rows={len(rfm):,} → {args.out}")
    print(rfm.head())


if __name__ == "__main__":
    main()
