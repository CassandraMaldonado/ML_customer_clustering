import os

file_path = "Models_Final.ipynb"
# Convert the Jupyter Notebook to a Python script
os.system(f"jupyter nbconvert --to script {file_path}")

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')

import streamlit as st
import plotly.express as px
import pandas as pd

# 🎭 Define Customer Personas with Updated GMM Insights
persona_details = {
    "Low Engagement Browser": {
        "img": "https://cdn-icons-png.flaticon.com/512/163/163810.png",
        "recommendations": "Target re-engagement campaigns, retargeting ads, and first-time purchase incentives.",
        "expected_revenue": "$50K",
        "metrics": {
            "Recency": "High (long time since last purchase)",
            "Frequency": "Low (infrequent shopper)",
            "Monetary": "Low (small purchase amounts)",
            "Engagement_Score": "Low (minimal interactions)"
        }
    },
    "Moderate Value Loyal": {
        "img": "https://cdn-icons-png.flaticon.com/512/1904/1904425.png",
        "recommendations": "Enhance loyalty programs, offer personalized recommendations, and provide early-access deals.",
        "expected_revenue": "$120K",
        "metrics": {
            "Recency": "Medium (consistent purchase history)",
            "Frequency": "Moderate (repeat customer)",
            "Monetary": "Medium (average spend)",
            "Engagement_Score": "High (frequent website visits, email interactions)"
        }
    },
    "High-Value Premium": {
        "img": "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        "recommendations": "Offer VIP programs, premium concierge services, and high-end product bundles.",
        "expected_revenue": "$200K",
        "metrics": {
            "Recency": "Low (recent purchases)",
            "Frequency": "High (frequent shopper)",
            "Monetary": "Very High (premium spender)",
            "Engagement_Score": "Very High (brand-loyal)"
        }
    },
    "Recent Low-Value Explorer": {
        "img": "https://cdn-icons-png.flaticon.com/512/3444/3444721.png",
        "recommendations": "Guide new customers with onboarding sequences, showcase testimonials, and offer educational content.",
        "expected_revenue": "$75K",
        "metrics": {
            "Recency": "Very Low (very recent purchases)",
            "Frequency": "Low (few transactions so far)",
            "Monetary": "Low (initial low-value purchases)",
            "Engagement_Score": "Medium (interested but uncertain)"
        }
    },
    "Occasional Big Spender": {
        "img": "https://cdn-icons-png.flaticon.com/512/3135/3135762.png",
        "recommendations": "Highlight luxury offerings, provide concierge services, and create premium shopping experiences.",
        "expected_revenue": "$180K",
        "metrics": {
            "Recency": "Low (recent but infrequent purchases)",
            "Frequency": "Low (rare transactions)",
            "Monetary": "Very High (large transactions when purchasing)",
            "Engagement_Score": "Medium (engages selectively)"
        }
    }
}

# 🎯 Streamlit UI Setup
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("📊 Customer Segmentation Analysis")
st.write("This dashboard presents insights into Acme Inc.'s customer segmentation model using Gaussian Mixture Models (GMM).")

# 📊 Display persona details dynamically
selected_persona = st.selectbox("Select a Customer Segment", list(persona_details.keys()))

if selected_persona:
    persona = persona_details[selected_persona]
    st.image(persona["img"], width=100)
    st.subheader(selected_persona)
    st.write("**Marketing Recommendations:**", persona["recommendations"])
    st.write("**Expected Revenue Impact:**", persona["expected_revenue"])

    # Show persona metrics in a dataframe format
    metrics_df = pd.DataFrame(list(persona["metrics"].items()), columns=["Metric", "Value"])
    st.table(metrics_df)

st.write("### 📌 Business Impact")
st.write("- Personalized marketing campaigns based on segments can enhance conversions.")
st.write("- GMM segmentation provides **data-driven** customer targeting.")
st.write("- By implementing targeted strategies, Acme Inc. aims to increase revenue by **$500K** annually.")
