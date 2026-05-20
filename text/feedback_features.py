"""NLTK-based features from unstructured customer feedback."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    _SIA = SentimentIntensityAnalyzer()
except Exception:                 # noqa: BLE001
    _SIA = None

NEG_KEYWORDS = {"refund", "cancel", "broken", "slow", "frustrated", "angry", "waste", "bad", "terrible", "expensive"}


def _sentiment(text: str) -> float:
    if _SIA is None:
        # cheap fallback if NLTK lexicons aren't downloaded
        text_l = text.lower()
        pos = sum(w in text_l for w in ("great", "love", "happy", "excellent", "easy"))
        neg = sum(w in text_l for w in NEG_KEYWORDS)
        return (pos - neg) / max(1, pos + neg)
    return float(_SIA.polarity_scores(text)["compound"])


def _has_neg_keyword(text: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return bool(tokens & NEG_KEYWORDS)


def aggregate(feedback: pd.DataFrame) -> pd.DataFrame:
    feedback = feedback.copy()
    feedback["sentiment"] = feedback["feedback"].astype(str).map(_sentiment)
    feedback["neg_kw"] = feedback["feedback"].astype(str).map(_has_neg_keyword)

    agg = feedback.groupby("customer_id").agg(
        sentiment_mean=("sentiment", "mean"),
        sentiment_std=("sentiment", "std"),
        neg_keyword_flag=("neg_kw", "any"),
        n_feedback=("feedback", "count"),
    ).reset_index()
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0.0)
    agg["neg_keyword_flag"] = agg["neg_keyword_flag"].astype(int)
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feedback", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fb = pd.read_parquet(args.feedback)
    agg = aggregate(fb)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(args.out, index=False)
    print(f"Text features for {len(agg):,} customers → {args.out}")


if __name__ == "__main__":
    main()
