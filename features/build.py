"""Merge RFM + text features into the modelling matrix."""
from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "recency_days", "frequency", "monetary",
    "sentiment_mean", "sentiment_std", "neg_keyword_flag", "n_feedback",
]


def merge(rfm: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    df = rfm.merge(text, on="customer_id", how="left")
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)
    df["sentiment_std"]  = df["sentiment_std"].fillna(0.0)
    df["neg_keyword_flag"] = df["neg_keyword_flag"].fillna(0).astype(int)
    df["n_feedback"] = df["n_feedback"].fillna(0).astype(int)
    return df
