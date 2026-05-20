"""K-means clustering with elbow + silhouette diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from features.build import NUMERIC_FEATURES, merge


MARKETING_K_RANGE = range(4, 7)


def choose_k(X: np.ndarray, k_range=MARKETING_K_RANGE) -> tuple[int, dict]:
    diagnostics = {"k": [], "inertia": [], "silhouette": []}
    best_k, best_sil = list(k_range)[0], -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_) if k > 1 else 0.0
        diagnostics["k"].append(k)
        diagnostics["inertia"].append(float(km.inertia_))
        diagnostics["silhouette"].append(float(sil))
        if sil > best_sil:
            best_k, best_sil = k, sil
    return best_k, diagnostics


def cluster(rfm_path: str, text_path: str, out_path: str, k: int | None = None):
    rfm = pd.read_parquet(rfm_path)
    text = pd.read_parquet(text_path)
    df = merge(rfm, text)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[NUMERIC_FEATURES])

    if k is None:
        k, diag = choose_k(X)
        print("\n=== K diagnostics ===")
        for kk, ine, sil in zip(diag["k"], diag["inertia"], diag["silhouette"]):
            print(f"  k={kk}  inertia={ine:>12.1f}  silhouette={sil:.3f}")
        print(f"\n→ chose k={k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    df["segment_id"] = km.labels_

    # Label segments by RFM-string mode for marketer readability
    seg_summary = (
        df.groupby("segment_id")
        .agg(
            size=("customer_id", "count"),
            recency=("recency_days", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
            sentiment=("sentiment_mean", "mean"),
            most_common_rfm=("rfm_string", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )
    print("\n=== Segment summary ===")
    print(seg_summary.to_string(index=False))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    seg_summary.to_csv(Path(out_path).with_name("segment_summary.csv"), index=False)
    print(f"\nSegments → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rfm", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=None)
    args = p.parse_args()
    cluster(args.rfm, args.text, args.out, args.k)


if __name__ == "__main__":
    main()
