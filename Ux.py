import os

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')

import streamlit as st
import plotly.express as px
import pandas as pd

# Sample Cluster Data (Replace with actual data)
cluster_means = pd.DataFrame({
    "Recency": [14, 25, 39, 16],
    "Frequency": [5, 52, 10, 25],
    "Monetary": [0.49, 0.48, 0.50, 0.49],
    "Engagement_Score": [3.76, 3.75, 3.74, 3.74]
}, index=["Loyal Mid-Value Shopper", "Super-Frequent Budget Shopper", "Occasional High-Value Buyer", "Steady Mid-Spender"])

# 📌 Define Personas with Image Paths & Recommendations
persona_details = {
    "Loyal Mid-Value Shopper": {
        "img": "persona1.png",
        "recommendations": "Offer loyalty rewards and targeted upselling.",
        "expected_revenue": "$50K"
    },
    "Super-Frequent Budget Shopper": {
        "img": "persona2.png",
        "recommendations": "Provide discount bundles and frequent shopper deals.",
        "expected_revenue": "$75K"
    },
    "Occasional High-Value Buyer": {
        "img": "persona3.png",
        "recommendations": "Personalized offers for high-value transactions.",
        "expected_revenue": "$120K"
    },
    "Steady Mid-Spender": {
        "img": "persona4.png",
        "recommendations": "Encourage repeat purchases with mid-tier promotions.",
        "expected_revenue": "$60K"
    }
}

# 🎭 Display Customer Personas
st.header("🛍️ Customer Personas")
cols = st.columns(len(persona_details))  # Create columns dynamically

selected_persona = None
for i, (persona, details) in enumerate(persona_details.items()):
    with cols[i]:
        st.image(details["img"], width=150)
        if st.button(f"{persona}"):
            selected_persona = persona  # Store selected persona

# 📊 Show Persona Behavior Breakdown (if selected)
if selected_persona:
    st.subheader(f"📊 Show Persona Behavior Breakdown: {selected_persona}")
    
    # 🎯 Show Marketing Recommendations
    st.markdown(f"**📢 Recommended Strategy:** {persona_details[selected_persona]['recommendations']}")
    st.markdown(f"💰 **Expected Revenue Impact:** {persona_details[selected_persona]['expected_revenue']}")

    # 📈 Bar Chart of Key Behavior Metrics
    persona_behavior = cluster_means.loc[selected_persona]
    fig = px.bar(persona_behavior, 
                 x=persona_behavior.index, 
                 y=persona_behavior.values,
                 title=f"Customer Behavior Breakdown - {selected_persona}",
                 labels={"x": "Metrics", "y": "Value"},
                 color=persona_behavior.index)
    
    st.plotly_chart(fig)