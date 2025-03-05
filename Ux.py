import os

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')
# Check if the dataset exists before loading
file_path = "shopping_behavior_updated.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    st.error(f"⚠️ The dataset '{file_path}' was not found. Please upload the file or check the path.")

import streamlit as st
import plotly.express as px
import pandas as pd

# 🎭 Define Customer Personas with Online Image Links, Recommendations, and Expected Revenue
persona_details = {
    "Loyal High-Spenders": {
        "img": "https://cdn-icons-png.flaticon.com/512/163/163810.png",
        "recommendations": "Create VIP membership with special rewards and exclusive products.",
        "expected_revenue": "$200K",
        "metrics": {
            "Recency": "Low (frequent recent purchases)",
            "Frequency": "High (regular shopper)",
            "Monetary": "Very High (premium spender)",
            "Engagement_Score": "Very High (brand-loyal shopper)"
        }
    },
    "Discount Hunters": {
        "img": "https://cdn-icons-png.flaticon.com/512/3596/3596027.png",
        "recommendations": "Offer frequent sales, limited-time deals, and loyalty points.",
        "expected_revenue": "$30K",
        "metrics": {
            "Recency": "Medium (buys frequently but waits for deals)",
            "Frequency": "High (repeat customer but with discounts)",
            "Monetary": "Low (spends little per purchase)",
            "Engagement_Score": "Medium (engaged but price-sensitive)"
        }
    },
    "New Customers": {
        "img": "https://cdn-icons-png.flaticon.com/512/2920/2920316.png",
        "recommendations": "Provide onboarding campaigns and first-purchase discounts.",
        "expected_revenue": "$15K",
        "metrics": {
            "Recency": "High (new shopper)",
            "Frequency": "Low (few past purchases)",
            "Monetary": "Medium (varies by category)",
            "Engagement_Score": "Low (needs nurturing)"
        }
    },
    "Seasonal Shoppers": {
        "img": "https://cdn-icons-png.flaticon.com/512/3534/3534084.png",
        "recommendations": "Send pre-sale notifications and off-season discounts.",
        "expected_revenue": "$50K",
        "metrics": {
            "Recency": "Low (buys once per season)",
            "Frequency": "Low (not year-round shopper)",
            "Monetary": "Medium (seasonal spending spikes)",
            "Engagement_Score": "Medium (engages during specific periods)"
        }
    },
    "Category Enthusiasts": {
        "img": "https://cdn-icons-png.flaticon.com/512/892/892689.png",
        "recommendations": "Offer bundles, related product suggestions, and loyalty perks.",
        "expected_revenue": "$80K",
        "metrics": {
            "Recency": "Medium (consistent purchases in one category)",
            "Frequency": "Medium (repeat purchases in niche items)",
            "Monetary": "High (focused spending)",
            "Engagement_Score": "High (brand-loyal within category)"
        }
    },
    "Luxury Buyers": {
        "img": "https://cdn-icons-png.flaticon.com/512/4228/4228554.png",
        "recommendations": "Offer high-end collections, exclusive services, and premium packaging.",
        "expected_revenue": "$150K",
        "metrics": {
            "Recency": "Medium (buys infrequently but in large amounts)",
            "Frequency": "Low (rare shopper)",
            "Monetary": "Very High (luxury spender)",
            "Engagement_Score": "Medium (expects premium experience)"
        }
    },
    "Frequent Small-Spenders": {
        "img": "https://cdn-icons-png.flaticon.com/512/3135/3135776.png",
        "recommendations": "Introduce subscription models, bulk discounts, and add-on purchases.",
        "expected_revenue": "$40K",
        "metrics": {
            "Recency": "High (buys often)",
            "Frequency": "High (regular shopper)",
            "Monetary": "Low (small purchases each time)",
            "Engagement_Score": "Medium (habitual but low-value shopper)"
        }
    },
    "Impulse Buyers": {
        "img": "https://cdn-icons-png.flaticon.com/512/1157/1157109.png",
        "recommendations": "Encourage flash sales, time-limited offers, and personalized recommendations.",
        "expected_revenue": "$60K",
        "metrics": {
            "Recency": "Medium (spontaneous purchases)",
            "Frequency": "Medium (inconsistent but returns)",
            "Monetary": "Medium (varied spending habits)",
            "Engagement_Score": "High (responsive to promotions)"
        }
    },
    "Subscription-Based Buyers": {
        "img": "https://cdn-icons-png.flaticon.com/512/4825/4825556.png",
        "recommendations": "Offer membership perks, exclusive discounts, and auto-renewals.",
        "expected_revenue": "$90K",
        "metrics": {
            "Recency": "Low (regular purchases on a schedule)",
            "Frequency": "High (subscription model)",
            "Monetary": "Medium (consistent spending)",
            "Engagement_Score": "Very High (brand-loyal)"
        }
    },
    "Low-Engagement Customers": {
        "img": "https://cdn-icons-png.flaticon.com/512/1828/1828665.png",
        "recommendations": "Send re-engagement campaigns, discounts, and personalized outreach.",
        "expected_revenue": "$10K",
        "metrics": {
            "Recency": "Very High (long time since last purchase)",
            "Frequency": "Low (infrequent shopper)",
            "Monetary": "Low (spends very little)",
            "Engagement_Score": "Very Low (rarely interacts)"
        }
    }
}

# 🖥️ Streamlit UI for Displaying Personas
st.title("Customer Segmentation Insights")

# Dropdown to select customer persona
selected_persona = st.selectbox("Select a Customer Persona:", list(persona_details.keys()))

# Display persona details
persona = persona_details[selected_persona]
st.image(persona["img"], width=150)
st.subheader(selected_persona)
st.write(f"**Expected Revenue:** {persona['expected_revenue']}")
st.write(f"**Marketing Recommendations:** {persona['recommendations']}")

# Display key engagement metrics
st.write("### Engagement Metrics:")
for metric, value in persona["metrics"].items():
    st.write(f"🔹 **{metric}:** {value}")

# 📊 Cluster Distribution Chart
st.subheader("Cluster Size Distribution")
df = pd.read_csv("shopping_behavior_updated.csv")  # Load dataset
cluster_counts = df["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Customer Count"]
fig = px.bar(cluster_counts, x="Cluster", y="Customer Count", text="Customer Count",
             title="Number of Customers in Each Cluster", color="Customer Count",
             color_continuous_scale="viridis")
st.plotly_chart(fig)