# ML Customer Clustering

Developed a machine learning solution to identify meaningful customer segments based on behavior, demographics and preferences, so marketing teams can personalize campaigns, improve conversion and maximize ROI.

---

## Why this project exists

Most marketing programs treat customers as one audience. In reality, customers behave differently:
- some are high-value repeat buyers,
- others only respond to discounts,
- some browse a lot but rarely purchase,
- some are new and need onboarding nudges.

**Customer clustering** turns a large customer base into a small set of actionable segments so teams can:
- tailor messaging and offers,
- allocate budget more efficiently,
- track ROI by segment instead of guessing.

---

## Objectives

1. **Identify key customer segments** using clustering and optional classification for segment assignment.
2. **Optimize marketing personalization** with segment profiles and recommended actions.
3. **Measure business impact** by evaluating how segmentation improves conversion and ROI.

---

## What’s inside this repo

High-level repository structure (major folders):

- `EDA/`: exploratory analysis, distributions, correlations, missingness and feature understanding.
- `K Clustering/`: clustering experiments.
- `Models/` — trained models / pipelines / supporting code artifacts :contentReference[oaicite:6]{index=6}  
- `Customer_segmentation/` — end-to-end segmentation workflow (feature engineering → clustering → profiling) :contentReference[oaicite:7]{index=7}  
- `Deliverables/` — final outputs (slides, reports, summaries, charts) :contentReference[oaicite:8]{index=8}  
- `Streamlit/`: app to run segmentation and inspect segments.
- `Recommender system/`: recommender work connected to the segmentation experiment.
- `Old versions/`: previous iterations and experiments.

---

## Method overview

Typical pipeline used in customer clustering projects like this:

1. **Data preparation**
   - clean missing values / outliers
   - encode categoricals
   - scale numeric features (important for distance-based clustering)

2. **Feature engineering**
   - behavioral: frequency, recency, monetary value (RFM), browsing depth, discount usage, channel preference
   - demographics: age band, region, household attributes (if available)
   - product preferences: category mix, brand affinity, basket size (if available)

3. **Clustering**
   - baseline: K-Means (fast + interpretable)
   - model selection: elbow + silhouette + stability checks
   - sanity checks: segment sizes, separability, business sense

4. **Segment profiling + naming**
   - top distinguishing features per segment
   - “who they are”, “what they want”, “how to market to them”

5. **Operationalization**
   - save model + preprocessing pipeline
   - assign new customers to segments
   - track outcomes (conversion, AOV, retention) by segment

## Streamlit app

If your Streamlit folder contains the app entrypoint (commonly `app.py` or `streamlit_app.py`), you can run it like:

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate   # (Mac/Linux)
# .venv\Scripts\activate    # (Windows)

# Install dependencies
pip install -r requirements.txt

# 3) Run Streamlit
streamlit run Streamlit/app.py
