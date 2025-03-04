import os

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np 

# Streamlit App Title
st.title("Customer Segmentation Dashboard")

# Load Data (Modify this part if loading from a file)
@st.cache_data
def load_data():
    # Dummy Data Example (Replace with actual dataset)
    data = pd.DataFrame({
        'Customer ID': range(1, 101),
        'Recency': np.random.randint(1, 50, 100),
        'Frequency': np.random.randint(1, 20, 100),
        'Monetary': np.random.uniform(50, 500, 100),
        'Cluster': np.random.choice([0, 1, 2, 3], 100)
    })
    return data

df = load_data()

# Show raw data
if st.checkbox("Show Raw Data"):
    st.write(df)

# Cluster Distribution
st.subheader("Cluster Distribution")
fig = px.histogram(df, x="Cluster", title="Customer Segmentation Clusters", nbins=10)
st.plotly_chart(fig)

# Scatter Plot for Cluster Insights
st.subheader("Recency vs Frequency by Cluster")
fig2 = px.scatter(df, x="Recency", y="Frequency", color="Cluster", title="Recency vs Frequency")
st.plotly_chart(fig2)

# Radar Chart for Customer Profiles
st.subheader("Cluster Profiles")

cluster_means = df.groupby("Cluster").mean()

fig3, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

angles = pd.np.linspace(0, 2 * pd.np.pi, len(cluster_means.columns), endpoint=False).tolist()
for cluster in cluster_means.index:
    ax.plot(angles, cluster_means.loc[cluster], label=f"Cluster {cluster}")
    ax.fill(angles, cluster_means.loc[cluster], alpha=0.2)

ax.set_xticks(angles)
ax.set_xticklabels(cluster_means.columns)
ax.set_title("Customer Segments")
ax.legend()

st.pyplot(fig3)

st.write("### Summary")
st.write("""
This dashboard visualizes customer segments based on behavioral clustering.
Use the charts above to understand different customer types and adjust marketing strategies accordingly.
""")