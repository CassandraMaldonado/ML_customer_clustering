import os

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')

import streamlit as st
import plotly.express as px
import pandas as pd

# 🎭 Define Customer Personas with Online Image Links, Recommendations, and Expected Revenue
persona_details = {
    "Loyal Mid-Value Shopper": {
        "img": "https://cdn-icons-png.flaticon.com/512/2922/2922510.png",
        "recommendations": "Offer loyalty rewards and targeted upselling.",
        "expected_revenue": "$50K",
        "metrics": {
            "Recency": "Low (recent purchases)",
            "Frequency": "Low (occasional but consistent shopping)",
            "Monetary": "Moderate (balanced spending per purchase)",
            "Engagement_Score": "High (active and engaged shopper)"
        }
    },
    "Super-Frequent Budget Shopper": {
        "img": "https://cdn-icons-png.flaticon.com/512/2922/2922656.png",
        "recommendations": "Provide discount bundles and frequent shopper deals.",
        "expected_revenue": "$75K",
        "metrics": {
            "Recency": "Moderate (steady shopping pattern)",
            "Frequency": "Very High (most frequent shopper)",
            "Monetary": "Low (small spending per transaction)",
            "Engagement_Score": "Moderate (engaged but price-sensitive)"
        }
    },
    "Occasional High-Value Buyer": {
        "img": "https://cdn-icons-png.flaticon.com/512/2922/2922529.png",
        "recommendations": "Personalized offers for high-value transactions.",
        "expected_revenue": "$120K",
        "metrics": {
            "Recency": "Very High (hasn't purchased in a long time)",
            "Frequency": "Low (rare but strategic purchases)",
            "Monetary": "Very High (big spender per transaction)",
            "Engagement_Score": "Low (not highly engaged but valuable)"
        }
    },
    "Steady Mid-Spender": {
        "img": "https://cdn-icons-png.flaticon.com/512/2922/2922591.png",
        "recommendations": "Encourage repeat purchases with mid-tier promotions.",
        "expected_revenue": "$60K",
        "metrics": {
            "Recency": "Moderate (regular shopper, not recent)",
            "Frequency": "Moderate (balanced number of purchases)",
            "Monetary": "Moderate (average spending per transaction)",
            "Engagement_Score": "Moderate (interacts but isn't highly engaged)"
        }
    }
}

# 📊 Sample Cluster Data (Replace with actual data)
cluster_means = pd.DataFrame({
    "Recency": [14, 25, 39, 16],
    "Frequency": [5, 52, 10, 25],
    "Monetary": [0.49, 0.48, 0.50, 0.49],
    "Engagement_Score": [3.76, 3.75, 3.74, 3.74]
}, index=["Loyal Mid-Value Shopper", "Super-Frequent Budget Shopper", "Occasional High-Value Buyer", "Steady Mid-Spender"])

# 🎭 Display Customer Personas as Clickable Cards
st.header("🛍️ Customer Personas")

cols = st.columns(len(persona_details))  # Create a column layout for persona selection

selected_persona = None  # Store the selected persona

for i, (persona, details) in enumerate(persona_details.items()):
    with cols[i]:
        st.image(details["img"], width=120)  # Display persona image
        if st.button(persona):
            selected_persona = persona  # Store the selected persona

# 📊 Show Persona Behavior Breakdown (if a persona is selected)
if selected_persona:
    st.subheader(f"📊 Show Persona Behavior Breakdown: {selected_persona}")

    # 🎯 Show Marketing Recommendations
    st.markdown(f"**📢 Recommended Strategy:** {persona_details[selected_persona]['recommendations']}")
    st.markdown(f"💰 **Expected Revenue Impact:** {persona_details[selected_persona]['expected_revenue']}")

    # 📊 Display Exact Metric Interpretation
    st.markdown("### 🔍 Customer Behavior Breakdown:")
    for metric, value in persona_details[selected_persona]["metrics"].items():
        st.markdown(f"- **{metric}:** {value}")

    # 📈 Bar Chart of Key Behavior Metrics
    persona_behavior = cluster_means.loc[selected_persona]
    fig = px.bar(persona_behavior, 
                 x=persona_behavior.index, 
                 y=persona_behavior.values,
                 title=f"Customer Behavior Breakdown - {selected_persona}",
                 labels={"x": "Metrics", "y": "Value"},
                 color=persona_behavior.index)

    st.plotly_chart(fig)

    # 📖 Explanation of Metrics
    st.markdown("### ℹ️ Understanding Customer Behavior Metrics")
    behavior_explanations = {
        "Recency": "📅 **Recency**: Measures how recently the customer made a purchase. Lower values indicate recent purchases, while higher values suggest longer gaps since the last purchase.",
        "Frequency": "🔄 **Frequency**: Counts how often the customer makes purchases. Higher values suggest repeat shoppers who make frequent transactions.",
        "Monetary": "💰 **Monetary Value**: Represents the average transaction amount. Higher values indicate high-spending customers.",
        "Engagement_Score": "📈 **Engagement Score**: Captures interaction levels based on purchases, reviews, and engagement with the brand. A higher score suggests a more engaged customer."
    }

    for metric, explanation in behavior_explanations.items():
        st.markdown(f"- {explanation}")