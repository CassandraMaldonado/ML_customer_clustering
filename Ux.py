import os

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Load Data (Simulated) ---
@st.cache_data
def load_data():
    return pd.DataFrame({
        'Cluster': [0, 1, 2, 3],
        'Persona': ["Loyal Mid-Value Shopper", "Super-Frequent Budget Shopper",
                    "Steady Mid-Spender", "Occasional High-Value Buyer"],
        'Image': [
            "https://cdn-icons-png.flaticon.com/512/2922/2922510.png",
            "https://cdn-icons-png.flaticon.com/512/2922/2922656.png",
            "https://cdn-icons-png.flaticon.com/512/2922/2922529.png",
            "https://cdn-icons-png.flaticon.com/512/2922/2922591.png"
        ],
        'Recommended Strategy': [
            "Send premium discounts & loyalty rewards.",
            "Increase purchase frequency with flash sales.",
            "Target with mid-tier product promotions.",
            "Offer high-end premium product bundles."
        ],
        'Estimated Revenue Impact ($)': [150000, 100000, 125000, 125000]
    })

df = load_data()

# --- Title ---
st.title("📊 Customer Segmentation Dashboard")

# --- Display Cluster Personas ---
st.subheader("👥 Customer Personas")
cols = st.columns(len(df))  # Create columns for each persona

for i, row in df.iterrows():
    with cols[i]:
        st.image(row['Image'], width=100)
        if st.button(row["Persona"]):
            st.session_state['selected_persona'] = row['Persona']

# --- Display Selected Persona Details ---
if "selected_persona" in st.session_state:
    selected_row = df[df["Persona"] == st.session_state['selected_persona']].iloc[0]
    
    st.markdown(f"### **🔍 Insights: {selected_row['Persona']}**")
    st.write(f"📌 **Recommended Strategy:** {selected_row['Recommended Strategy']}")
    st.write(f"💰 **Estimated Revenue Impact:** ${selected_row['Estimated Revenue Impact ($)']:,.2f}")

    with st.expander("📊 Show Persona Behavior Breakdown"):
        st.write("Detailed customer behavior insights go here...")

# --- Show Cluster Profiles as a Radar Chart ---
st.subheader("📈 Cluster Profiles")
cluster_means = pd.DataFrame({
    "Recency": np.random.randint(5, 50, 4),
    "Frequency": np.random.randint(5, 60, 4),
    "Monetary": np.random.uniform(0.3, 0.7, 4),
    "Engagement Score": np.random.uniform(3.5, 4.0, 4)
}, index=df["Persona"])

angles = np.linspace(0, 2 * np.pi, len(cluster_means.columns), endpoint=False)
fig = px.line_polar(cluster_means.T, r=cluster_means.values.T, theta=cluster_means.columns, 
                    line_close=True, title="Cluster Radar Chart")

st.plotly_chart(fig)