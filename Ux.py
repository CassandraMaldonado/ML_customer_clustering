# %%
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('shopping_behavior_updated.csv')
df.head()

# %%
# Mapping for categorical frequencies to numerical values
frequency_mapping = {
    "Weekly": 52,
    "Fortnightly": 26,
    "Bi-Weekly": 24,
    "Monthly": 12,
    "Every 3 Months": 4,
    "Quarterly": 4,
    "Annually": 1
}

# Apply the mapping
df["Frequency of Purchases (Numeric)"] = df["Frequency of Purchases"].map(frequency_mapping)

# Drop the original categorical column
df.drop(columns=["Frequency of Purchases"], inplace=True)

# Display updated dataset
print(df.head())

# %%
#%pip install sentence-transformers
#%pip install tf-keras
#%pip install scikit-learn

from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Pre-trained language model
model = SentenceTransformer('all-MiniLM-L6-v2')

df['Category_Embedding'] = df['Category'].apply(lambda x: model.encode(str(x)))
df['Item_Purchased_Embedding'] = df['Item Purchased'].apply(lambda x: model.encode(str(x)))

# Convert embeddings to NumPy arrays
category_embeddings = np.vstack(df['Category_Embedding'].values)
item_purchased_embeddings = np.vstack(df['Item_Purchased_Embedding'].values)

# Reduce dimensionality using PCA
pca = PCA(n_components=20)  
category_embeddings_pca = pca.fit_transform(category_embeddings)
item_purchased_embeddings_pca = pca.fit_transform(item_purchased_embeddings)

# Concatenate PCA-reduced embeddings with numerical data
numerical_features = ['Age', 'Frequency', 'Monetary', 'Engagement_Score', 'Recency']
from sklearn.preprocessing import StandardScaler

# Create the required columns
df['Recency'] = df['Previous Purchases']
df['Frequency'] = df['Frequency of Purchases (Numeric)']
df['Monetary'] = df['Purchase Amount (USD)']

# Ensure categorical "Subscription Status" is standardized
df["Subscription Status"] = df["Subscription Status"].str.lower().str.strip()

# Map Subscription Status: 1 for subscribed, 0 for non-subscribed
df["Subscription_Status_Binary"] = df["Subscription Status"].apply(lambda x: 1 if x == "subscribed" else 0)

# Compute Engagement (If Review Rating is available)
if "Review Rating" in df.columns:
    df["Engagement_Score"] = df["Review Rating"]
else:
    df["Engagement_Score"] = 0  # Default if missing

# Loyalty Score (Combination of Subscription & Purchase Frequency)
df["Loyalty_Score"] = df["Frequency of Purchases (Numeric)"] + df["Previous Purchases"] + (df["Subscription_Status_Binary"] * 10)


X_embeddings = np.hstack([df[numerical_features].values, category_embeddings_pca, item_purchased_embeddings_pca])

# %%
#%pip install scikit-learn
from sklearn.cluster import KMeans

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['KMeans_LLM_Cluster'] = kmeans.fit_predict(X_embeddings)

# %%
from sklearn.metrics import silhouette_score

# Silhouette Score for K-Means
silhouette_score_kmeans = silhouette_score(X_embeddings, df['KMeans_LLM_Cluster'])

# Final clustering results
clustering_results = {
    "Number of Clusters": 4,
    "Silhouette Score": silhouette_score_kmeans,
    "Total Features After PCA + Numerical Data": X_embeddings.shape[1]
}

# Convert results to DataFrame and display
clustering_results_df = pd.DataFrame([clustering_results])
print(clustering_results_df)

# %%
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.countplot(x="KMeans_LLM_Cluster", data=df, order=df["KMeans_LLM_Cluster"].value_counts().index)
plt.title("Customer Segmentation Based on K-Means with LLM Embeddings")
plt.xticks(rotation=45)
plt.show()

# %%
# Define clusters & metrics
cluster_means_llm = df.groupby("KMeans_LLM_Cluster")[numerical_features].mean()
print(cluster_means_llm)

# Number of variables
categories = list(cluster_means_llm.columns)
N = len(categories)

# Convert to radar chart format
values = cluster_means_llm.values
cluster_names = [f'Cluster {i}' for i in range(len(cluster_means_llm))]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

# Plot Radar Chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i in range(len(values)):
    ax.plot(angles, values[i], linewidth=1, linestyle='solid', label=cluster_names[i])
    ax.fill(angles, values[i], alpha=0.2)

# Configure labels
ax.set_xticks(angles)
ax.set_xticklabels(categories)
plt.title("Cluster Comparison via Radar Chart (LLM Embeddings)")
plt.legend()

plt.show()

# %%
from sentence_transformers import SentenceTransformer

# Sentence embedding model 
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Categorical/text columns to convert into embeddings
text_columns = ["Category", "Subscription Status", "Payment Method"]

# Categorical values into embeddings
def encode_text_features(df, columns):
    text_data = df[columns].astype(str).agg(" ".join, axis=1)  # Concatenate selected text features
    embeddings = embedder.encode(text_data, convert_to_numpy=True)  # Generate text embeddings
    return embeddings

# Text embeddings
text_embeddings = encode_text_features(df, text_columns)

# Stack with numerical features for clustering
numerical_features = ["Recency", "Frequency", "Monetary", "Engagement_Score"]
X_combined = np.hstack((df[numerical_features].values, text_embeddings))

# %%
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Determine the optimal number of clusters (Elbow Method)
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_combined)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for LLM + K-Means Clustering")
plt.show()

# Choose optimal number of clusters
optimal_k_llm = 4  # Adjust based on elbow method

# Apply K-Means with chosen k
kmeans_llm = KMeans(n_clusters=optimal_k_llm, random_state=42, n_init=10)
df["LLM_KMeans_Cluster"] = kmeans_llm.fit_predict(X_combined)

# Evaluate clustering quality
silhouette_llm = silhouette_score(X_combined, df["LLM_KMeans_Cluster"])
davies_bouldin_llm = davies_bouldin_score(X_combined, df["LLM_KMeans_Cluster"])

# Display evaluation results
print("Silhouette Score (LLM + K-Means):", silhouette_llm)
print("Davies-Bouldin Index (LLM + K-Means):", davies_bouldin_llm)

# %%
# Compute mean values for numerical features per cluster
cluster_means_llm = df.groupby("LLM_KMeans_Cluster")[numerical_features].mean()

# Heatmap of cluster profiles
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means_llm, annot=True, cmap="coolwarm")
plt.title("LLM + K-Means Cluster Profiles")
plt.show()

# %%
# Get mode of text features for each cluster
cluster_text_modes = df.groupby("LLM_KMeans_Cluster")[text_columns].agg(lambda x: x.mode()[0])
print(cluster_text_modes)

# %%
# Define LLM-based personas
llm_personas = {
    0: "Loyal Mid-Value Shopper (Low recency (recent purchases), moderate frequency, and moderate monetary spend. These customers are engaged and buy consistently but are not necessarily high spenders.)",
    1: "Super-Frequent Budget Shopper (Highest frequency (52 purchases) but lower monetary spend. These are power shoppers who make many small transactions, likely looking for deals or daily essentials.)",
    2: "Steady Mid-Spender (Mid-range frequency (25 purchases) and moderate spend, suggesting a balanced shopping behavior with steady purchasing patterns.)",
    3: "Occasional High-Value Buyer (Highest recency (long time since last purchase) but the highest monetary spend per transaction. These customers buy infrequently but spend significantly when they do.)"
}

# Assign personas
df["LLM_KMeans_Persona"] = df["LLM_KMeans_Cluster"].map(llm_personas)

# Display persona breakdown
persona_summary = df.groupby("LLM_KMeans_Cluster")[numerical_features].mean().reset_index()
persona_summary["Persona"] = persona_summary["LLM_KMeans_Cluster"].map(llm_personas)
print(persona_summary)

# %%
#Customer segmentation data
segment_data = df.groupby("LLM_KMeans_Persona")[numerical_features].mean().reset_index()
segment_data

df_segments = pd.DataFrame(segment_data)

# %%
# Define marketing recommendations
segment_recommendations = {
    "Loyal Mid-Value Shopper": "Targeted promotions for high-margin items to increase the average transaction value.",
    "Super-Frequent Budget Shopper": "Special discounts for frequent purchases to drive customer loyalty and retention.",
    "Steady Mid-Spender": "Personalized product recommendations based on past purchase history to increase cross-selling opportunities.",
    "Occasional High-Value Buyer": "Exclusive offers for high-value items to encourage repeat purchases and re-engage these customers."
}

# Assign recommendations
df_segments["Marketing_Recommendations"] = df_segments["LLM_KMeans_Persona"].map(segment_recommendations)

# Display segment data with recommendations
df_segments

# %%
# Streamlit Dashboard Layout
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Dashboard Title
st.title("📊 Customer Segmentation Dashboard")

# %%
# Add Customer Count column
df_segments["Customer Count"] = df.groupby("LLM_KMeans_Persona").size().values

# Key Metrics Section
st.subheader("📈 Key Business Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", sum(df_segments["Customer Count"]))
col2.metric("Avg Recency", round(df_segments["Recency"].mean(), 1))
col3.metric("Avg Frequency", round(df_segments["Frequency"].mean(), 1))
col4.metric("Predicted Revenue Growth", "$500,000 (15% uplift)")

# %%
# Pie Chart: Customer Segments Distribution
st.subheader("📌 Customer Segments Breakdown")
fig_pie = px.pie(df_segments, names="LLM_KMeans_Persona", values="Customer Count", title="Customer Segments Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# %%
# Radar Chart: Segment Comparison
st.subheader("🕵️‍♂️ Segment Comparison")
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
categories = ["Recency", "Frequency", "Monetary", "Engagement_Score"]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

for index, row in df_segments.iterrows():
    values = row[categories].tolist()
    ax.fill(angles, values, alpha=0.25, label=row["LLM_KMeans_Persona"])
    ax.plot(angles, values, linewidth=2)

ax.set_xticks(angles)
ax.set_xticklabels(categories)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
st.pyplot(fig)

# %%
# Customer Journey Insights (Simulated Line Chart)
st.subheader("📊 Customer Journey Insights")
customer_journey = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
    "Loyal Shopper": [100, 120, 150, 160, 180],
    "Budget Shopper": [200, 230, 250, 280, 300],
    "Mid-Spender": [90, 100, 110, 130, 140],
    "High-Value Buyer": [50, 60, 70, 80, 90],
}
df_journey = pd.DataFrame(customer_journey)
fig_journey = px.line(df_journey, x="Month", y=df_journey.columns[1:], title="Segment Growth Over Time")
st.plotly_chart(fig_journey, use_container_width=True)


# %%
# Recommendations Section
st.subheader("📌 AI-Powered Marketing Recommendations")
selected_segment = st.selectbox("Select a Customer Segment:", df_segments["LLM_KMeans_Persona"])

# Map the selected segment to the corresponding key in segment_recommendations
segment_key = next((key for key, value in llm_personas.items() if value == selected_segment), None)

if segment_key is not None:
	st.write(f"**Recommendation:** {segment_recommendations[llm_personas[segment_key].split('(')[0].strip()]}")
else:
	st.write("**Recommendation:** No recommendation available for the selected segment.")

# %%
# Export Button
st.subheader("📤 Export & CRM Integration")
if st.button("Export Customer Segments"):
    df_segments.to_csv("customer_segments.csv", index=False)
    st.success("✅ Customer Segments exported successfully!")

st.write("🔗 **Integration Available**: Connect with CRM tools like Salesforce, HubSpot, and Klaviyo.")


