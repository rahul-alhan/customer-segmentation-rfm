"""Stakeholder-friendly segment explorer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation — RFM + Sentiment")

seg_path = Path("data/segments.parquet")
if not seg_path.exists():
    st.error("Run the pipeline first — see README quickstart.")
    st.stop()

df = pd.read_parquet(seg_path)
st.sidebar.metric("Customers", f"{len(df):,}")
st.sidebar.metric("Segments", df["segment_id"].nunique())

st.subheader("Segment sizes")
sizes = df.groupby("segment_id").size().reset_index(name="n")
st.plotly_chart(px.bar(sizes, x="segment_id", y="n"), use_container_width=True)

st.subheader("Recency vs Monetary, coloured by segment")
fig = px.scatter(
    df, x="recency_days", y="monetary", color="segment_id",
    hover_data=["customer_id", "rfm_string", "sentiment_mean"],
    height=480,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sentiment by segment")
fig2 = px.box(df, x="segment_id", y="sentiment_mean", points="suspectedoutliers")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Segment profiles")
st.dataframe(
    df.groupby("segment_id").agg(
        size=("customer_id", "count"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_sentiment=("sentiment_mean", "mean"),
        pct_negative_kw=("neg_keyword_flag", "mean"),
    ).round(2),
    use_container_width=True,
)
