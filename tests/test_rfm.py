"""Smoke tests for RFM scoring + text aggregation."""
from __future__ import annotations

import pandas as pd

from features.rfm import compute_rfm
from text.feedback_features import aggregate
from scripts.generate_data import generate


def test_rfm_columns_present():
    txn, _ = generate(200)
    rfm = compute_rfm(txn)
    for c in ["customer_id", "recency_days", "frequency", "monetary", "R", "F", "M", "rfm_string"]:
        assert c in rfm.columns


def test_rfm_string_format():
    txn, _ = generate(150)
    rfm = compute_rfm(txn)
    sample = rfm["rfm_string"].iloc[0]
    parts = sample.split("_")
    assert len(parts) == 3 and parts[0].startswith("R") and parts[1].startswith("F") and parts[2].startswith("M")


def test_text_aggregate_handles_empty():
    fb = pd.DataFrame({"customer_id": ["c_1"], "feedback": ["This is great!"], "created_at": [pd.Timestamp("2024-01-01")]})
    agg = aggregate(fb)
    assert len(agg) == 1
    assert "sentiment_mean" in agg.columns
