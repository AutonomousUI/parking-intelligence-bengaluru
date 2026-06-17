# AI-Driven Parking Intelligence: Detecting Hotspots & Quantifying Congestion Impact

**Gridlock Hackathon 2.0 — Round 2 Submission**
Theme: *Poor Visibility on Parking-Induced Congestion*

## Problem

Traffic enforcement in Bengaluru is reactive and patrol-based. There is no system to identify which illegal parking violations actually cause congestion versus which are harmless. Officers cannot prioritize where to deploy limited patrol resources.

## Solution

A 5-stage AI pipeline that transforms raw parking violation data into actionable enforcement intelligence:

1. **Hotspot Detection (DBSCAN)** — Clusters 298,445 real violation records into 157 physical hotspot zones using GPS coordinates, covering 99.7% of all violations.
2. **Feature Engineering** — Cyclical time encoding (sin/cos) for hour-of-day patterns, Haversine distance to city center, peak-hour flags, and zone-level aggregation statistics.
3. **Congestion Risk Score (CRS)** — A weighted 0–100 score combining violation density, peak-hour fraction, proximity to center, recurrence, and weekend activity.
4. **LightGBM Scoring Model** — Trained to predict CRS for every zone, achieving R² = 0.862 and RMSE = 3.39 on validation data.
5. **Enforcement Priority Map** — An interactive Folium heatmap and ranked CSV output, color-coded Red/Orange/Yellow/Green, so dispatchers can immediately identify where to deploy patrol units.

## Key Results

| Metric | Value |
|---|---|
| Violations analysed | 298,445 |
| Date range | Nov 2023 – Apr 2024 |
| Hotspot clusters found | 157 |
| Coverage | 99.7% of violations |
| Model R² | 0.862 |
| Model RMSE | 3.39 |
| HIGH priority zones | 3 |
| MEDIUM priority zones | 81 |
| LOW priority zones | 73 |
| Peak violation hour | 5:00 AM |
| Peak violation day | Sunday |

## Repository Contents

- `grid2_notebook.ipynb` — Full Jupyter notebook with data loading, cleaning, EDA, feature engineering, DBSCAN clustering, LightGBM model training, and output generation.
- `AI_Parking_Intelligence_Presentation.pptx` — Pitch deck covering problem, solution, methodology, and results.

## How to Run

**Option A — Kaggle (recommended)**
Live demo: https://www.kaggle.com/code/swapnomonmurari/grid2
Click "Copy & Edit", attach the dataset under "Add Input", then "Run All".

**Option B — Locally / Google Colab**

```bash
pip install pandas numpy scikit-learn lightgbm folium matplotlib seaborn
```

1. Place the violation CSV file in the working directory.
2. Open `grid2_notebook.ipynb`.
3. Update the `DATA_PATH` variable if your file path differs.
4. Run all cells top to bottom.

**Outputs generated:**
- `enforcement_priority_zones.csv` — ranked zone list with Congestion Risk Scores
- `parking_enforcement_map.html` — interactive heatmap (open in any browser)

Total runtime: ~3–5 minutes on Kaggle's free CPU tier.

## Tech Stack

Python 3.12 · Pandas · NumPy · Scikit-learn (DBSCAN) · LightGBM · Folium · Matplotlib · Seaborn

## Why It Matters

This shifts enforcement from reactive to proactive. Instead of officers responding to violations after gridlock has formed, dispatchers can pre-position units in high-risk zones before rush hour begins — using real data instead of intuition.
