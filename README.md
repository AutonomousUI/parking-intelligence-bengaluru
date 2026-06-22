# AI-Driven Parking Intelligence: Detecting Hotspots & Quantifying Congestion Impact

**Gridlock Hackathon 2.0 — Round 2 Submission**
Theme: *Poor Visibility on Parking-Induced Congestion*

## Live Demo

**Interactive Dashboard:** Run locally with `streamlit run app.py` (see setup below)
**Kaggle Notebook (full pipeline, pre-executed):** https://www.kaggle.com/code/swapnomonmurari/grid2

## One File You Need to Add

This repo does not include the raw violation dataset — it's the official file provided by the hackathon, and not ours to redistribute. To run `app.py` or re-run the notebook from scratch, place your own copy of the officially-provided file (named something like `jan to may police violation_anonymized791b166.csv`) in the project folder first. The app auto-detects the dataset by filename pattern, so exact naming doesn't matter.

Judges already have this file from HackerEarth/BTP. Browsing this repo, reading the notebook's saved outputs, or watching the demo video requires no extra files at all.

## Problem

Traffic enforcement in Bengaluru is reactive and patrol-based. There is no system to identify which illegal parking violations actually cause congestion versus which are harmless, so officers cannot prioritize where to deploy limited patrol resources.

## Solution

A 5-stage AI pipeline that turns raw parking violation data into actionable enforcement intelligence, presented through an interactive Streamlit dashboard:

1. **Hotspot Detection (DBSCAN)** — clusters 298,450 real violation records into 157 physical hotspot zones using GPS coordinates, covering 99.7% of all violations.
2. **Feature Engineering** — cyclical time encoding (sin/cos) for hour-of-day patterns, Haversine distance to city center, peak-hour flags, and zone-level aggregation statistics.
3. **Congestion Risk Score (CRS)** — a weighted 0–100 score combining violation density, peak-hour fraction, proximity to center, recurrence, and weekend activity. Weights are domain-informed, not fitted to a labelled outcome — see Limitations below.
4. **LightGBM Scoring Model** — learns nonlinear relationships between zone characteristics to refine the ranking (R² = 0.862, RMSE = 3.39 against the engineered CRS target). A feature-importance breakdown is shown directly in the dashboard.
5. **Interactive Dashboard** — a 6-page Streamlit app: Executive Overview, Hotspot Map, Enforcement Priority Table, Zone Explorer, Trend Analytics, and an Impact Simulator, so dispatchers can explore and act on results without touching code.

## Key Results

| Metric | Value |
|---|---|
| Violations analysed | 298,450 |
| Date range | Nov 2023 – Apr 2024 |
| Hotspot clusters found | 157 |
| Coverage | 99.7% of violations |
| Model R² (vs. engineered CRS) | 0.862 |
| Model RMSE | 3.39 |
| HIGH priority zones | 3 |
| MEDIUM priority zones | 81 |
| LOW priority zones | 73 |
| Peak violation hour | 5:00 AM |
| Peak violation day | Sunday |

## Repository Contents

- `app.py` — Streamlit dashboard (main deliverable)
- `enforcement_priority_zones.csv` — pre-computed zone-level output (157 zones, all features, CRS, rank, and tier)
- `grid2_notebook.ipynb` — full Jupyter notebook covering data loading, cleaning, EDA, feature engineering, DBSCAN, LightGBM training, and output generation
- `AI_Parking_Intelligence_Presentation.pptx` — pitch deck covering problem, solution, methodology, and results
- `requirements.txt` — Python dependencies for the dashboard

## How to Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app needs two CSVs in the same folder:
- `enforcement_priority_zones.csv` — included in this repo
- The official raw violation dataset (see "One File You Need to Add" above) — not included here, per competition rules. The app detects it automatically by filename pattern, so it doesn't need to be renamed.

It opens automatically at `http://localhost:8501`.

## How to Run the Full Pipeline (Notebook)

**Option A — Kaggle (recommended, dataset pre-attached)**
https://www.kaggle.com/code/swapnomonmurari/grid2 → "Copy & Edit" → "Run All"

**Option B — Locally or in Colab**
```bash
pip install pandas numpy scikit-learn lightgbm folium matplotlib seaborn
```
Place the official violation CSV in the working directory, update `DATA_PATH` if needed, then run `grid2_notebook.ipynb` top to bottom (around 3–5 minutes, no GPU required).

## Tech Stack

Python 3.12 · Pandas · NumPy · Scikit-learn (DBSCAN) · LightGBM · Folium · Plotly · Streamlit

## Limitations & Honest Notes

- **CRS is an engineered proxy, not a measured ground truth.** No public dataset links these violations to real-time congestion. CRS weights reflect domain reasoning — density, timing, and proximity matter most — rather than a fitted label. LightGBM learns nonlinear relationships between zone features to support ranking, not to "predict" verified congestion.
- **The Impact Simulator page produces illustrative projections** based on the CRS formula, not validated real-world forecasts.
- **No real-time data feed yet.** See the Future Roadmap section in the presentation for planned phases: live violation feed integration, CCTV integration, and predictive alerts.

## Why It Matters

This shifts enforcement from reactive to proactive. Instead of officers responding to violations after gridlock has formed, dispatchers can pre-position units in high-risk zones before rush hour begins — using real data and a live dashboard instead of intuition.
