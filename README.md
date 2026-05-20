# Customer Segmentation — RFM + K-means + NLTK Feedback Augmentation

Customer segmentation that combines classical **RFM scoring** with **K-means clustering**, augmented by **NLTK-derived signals** from unstructured customer feedback. Outputs are wrapped in a **Streamlit dashboard** for non-technical stakeholders.

> Mirrors a segmentation system I built in a prior role (2021–2022) — improved marketing-campaign efficiency by ~22%.

---

## Why This Repo Exists

Most "customer segmentation" tutorials stop at RFM percentiles. Real marketing teams need:

1. **Behavioral truth** (RFM)
2. **Statistical clustering** (K-means with elbow + silhouette diagnostics)
3. **Voice-of-customer signal** (NLTK on free-text feedback) — this is the part that turned RFM percentile-buckets into segments marketers actually trusted

This repo shows all three working together.

---

## Pipeline

```
   transactions  ──▶  rfm.py
                          │ (Recency, Frequency, Monetary scoring)
                          ▼
   customer feedback  ─▶  text/feedback_features.py
                          │ (NLTK preprocessing + sentiment + keywords)
                          ▼
                     features/build.py
                          │ (RFM + text features merged)
                          ▼
                     clustering/kmeans.py
                          │ (k chosen via elbow + silhouette)
                          ▼
                     segments.parquet
                          │
                          ▼
                  dashboard/streamlit_app.py
                  (segment profiles + drilldowns)
```

---

## Quickstart

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords vader_lexicon punkt

# 1. Generate synthetic transactions + feedback
python -m scripts.generate_data --customers 2000 --out data/

# 2. Compute RFM
python -m features.rfm --transactions data/transactions.parquet --out data/rfm.parquet

# 3. Build text features from feedback
python -m text.feedback_features --feedback data/feedback.parquet --out data/text_features.parquet

# 4. Merge + cluster
python -m clustering.kmeans \
  --rfm data/rfm.parquet \
  --text data/text_features.parquet \
  --out data/segments.parquet

# 5. Launch dashboard
streamlit run dashboard/streamlit_app.py
```

---

## RFM Scoring

Each customer is scored on three axes:

| Axis | What it measures |
|---|---|
| **Recency** | Days since last purchase. Lower = better. |
| **Frequency** | Number of purchases over the lookback window. Higher = better. |
| **Monetary** | Total spend over the lookback window. Higher = better. |

Each axis is bucketed into 1–5 quintiles, producing an RFM-string like `R5_F4_M5`. Marketing already speaks this language — it's worth keeping alongside the cluster IDs.

---

## NLTK Feedback Features

For each customer with at least one feedback record:

- **Sentiment** — VADER compound score (mean across their feedback)
- **Sentiment volatility** — std-dev of compound score
- **Keyword density** — TF-IDF top-K terms aggregated per customer
- **Negative keyword flag** — boolean: any feedback contains words from a domain-tuned negative list (`"refund"`, `"cancel"`, `"slow"`, ...)

These features get **merged into the RFM table**, so K-means clusters on the combined signal — segments end up reflecting *both* what customers do *and* what they say.

---

## Choosing K

K is chosen by combining:

- **Elbow method** on inertia (find the bend)
- **Silhouette score** for K in [2..8]
- **Marketing constraint**: prefer K between 4 and 6 (more segments → harder to action)

The CLI prints both diagnostics and picks the K with the highest silhouette inside the marketing range. Override with `--k`.

---

## Repository Layout

```
customer-segmentation-rfm/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   └── generate_data.py
├── features/
│   ├── __init__.py
│   ├── rfm.py
│   └── build.py
├── text/
│   ├── __init__.py
│   └── feedback_features.py
├── clustering/
│   ├── __init__.py
│   └── kmeans.py
├── dashboard/
│   └── streamlit_app.py
└── tests/
    └── test_rfm.py
```

---

## Outcomes (original deployment)

- **~22%** lift in marketing-campaign efficiency (response rate × revenue per send)
- Marketers stopped asking *"which cluster is which?"* — the RFM-string + sentiment summary in the dashboard made segments self-explanatory
- The text-feature integration was the **first NLP layer ever added** to the company's classical ML pipeline

---

## License

MIT
