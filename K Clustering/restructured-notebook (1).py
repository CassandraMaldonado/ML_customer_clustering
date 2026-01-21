import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    import hdbscan
    from kmodes.kprototypes import KPrototypes
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Some advanced packages are not installed. You may need to install them with pip.")


# Initial exploration.

df_original = pd.read_csv('shopping_behavior_updated.csv')
df = df_original.copy()  # Create a working copy of the original data

# Display first few rows
print(df.head())

# Get basic information
print(df.info())

# Generate descriptive statistics
print(df.describe())


# Base transformations.


# 3.1 Mapping categorical frequencies to numerical values
frequency_mapping = {
    "Weekly": 52,
    "Fortnightly": 26,
    "Bi-Weekly": 24,
    "Monthly": 12,
    "Every 3 Months": 4,
    "Quarterly": 4,
    "Annually": 1
}

# Create a new column with numeric frequency values
df["Frequency of Purchases (Numeric)"] = df["Frequency of Purchases"].map(frequency_mapping)

# 3.2 RFM base features that all models will need
df['Recency'] = df['Previous Purchases']
df['Frequency'] = df['Frequency of Purchases (Numeric)']
df['Monetary'] = df['Purchase Amount (USD)']

# 3.3 Engagement Score based on review rating
if "Review Rating" in df.columns:
    df["Engagement_Score"] = df["Review Rating"]
else:
    df["Engagement_Score"] = 0  # Default if missing

# 3.4 Binary subscription status
df["Subscription Status"] = df["Subscription Status"].str.lower().str.strip()
df["Subscription_Status_Binary"] = df["Subscription Status"].apply(lambda x: 1 if x == "subscribed" else 0)

# 3.5 Loyalty Score as a combination of metrics
df["Loyalty_Score"] = df["Frequency"] + df["Recency"] + (df["Subscription_Status_Binary"] * 10)

# Store the original DataFrame with basic feature engineering for future models
df_base = df.copy()

# -------------------------------------------
# Section 4: K-means Clustering Analysis
# -------------------------------------------

# 4.1 Create a separate copy for K-means modeling
df_kmeans = df_base.copy()

# 4.2 Feature preparation for K-means
# Categorical variables to lowercase for consistency
categorical_cols = ['Gender', 'Category']
for col in categorical_cols:
    if col in df_kmeans.columns:
        df_kmeans[col] = df_kmeans[col].str.lower().str.strip()

# 4.3 Feature selection and standardization for K-means
features_for_kmeans = ['Recency', 'Frequency', 'Monetary']
scaler_kmeans = StandardScaler()
df_kmeans_scaled = scaler_kmeans.fit_transform(df_kmeans[features_for_kmeans])

# 4.4 Find optimal K using Elbow Method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_kmeans_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# 4.5 Apply K-means with optimal k
optimal_k = 5  # Identified from elbow plot
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_model.fit_predict(df_kmeans_scaled)

# Add cluster labels to the dataframe
df_kmeans["KMeans_Cluster"] = cluster_labels

# 4.6 Analysis of K-means Clusters

# 4.6.1 Create a DataFrame with scaled features for analysis
df_kmeans_analysis = pd.DataFrame(df_kmeans_scaled, columns=features_for_kmeans)
df_kmeans_analysis["KMeans_Cluster"] = cluster_labels

# Calculate cluster means
cluster_means = df_kmeans_analysis.groupby("KMeans_Cluster").mean()
print("Cluster means (scaled features):")
print(cluster_means)

# 4.6.2 Evaluation metrics
silhouette = silhouette_score(df_kmeans_scaled, cluster_labels)
davies_bouldin = davies_bouldin_score(df_kmeans_scaled, cluster_labels)
inertia_final = kmeans_model.inertia_

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Inertia (WCSS): {inertia_final:.4f}")

# 4.6.3 Visualize cluster characteristics
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_means, annot=True, cmap="coolwarm")
plt.title("K-Means Cluster Centers (RFME Features)")
plt.show()

# 4.6.4 Distribution of spending by cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x="KMeans_Cluster", y="Monetary", data=df_kmeans)
plt.title("Spending Distribution Across Clusters")
plt.show()

# 4.7 Cluster Profiling
# Define personas based on cluster characteristics
kmeans_personas = {
    0: "High-Value Loyal Customer (High monetary spend with consistent frequency)",
    1: "Recent Engaged Shopper (Good engagement score with recent purchases)",
    2: "Frequent Low-Spender (High frequency but low monetary value per transaction)",
    3: "Infrequent Low-Spender (Low frequency and low monetary value)",
    4: "Dormant High-Value Customer (High recency, meaning they haven't purchased in a while)"
}

# Map personas to clusters
df_kmeans["KMeans_Persona"] = df_kmeans["KMeans_Cluster"].map(kmeans_personas)

# Summarize profiles with persona labels
cluster_profiles_kmeans = df_kmeans.groupby("KMeans_Cluster")[features_for_kmeans + ["Loyalty_Score", "Engagement_Score"]].mean().reset_index()
cluster_profiles_kmeans["Persona"] = cluster_profiles_kmeans["KMeans_Cluster"].map(kmeans_personas)

print("\nK-Means Cluster Profiles:")
print(cluster_profiles_kmeans)

# -------------------------------------------
# Section 5: Hierarchical Clustering Analysis
# -------------------------------------------

# 5.1 Create a separate copy for hierarchical clustering
df_hierarchical = df_base.copy()

# 5.2 Feature selection for hierarchical clustering 
features_hierarchical = ['Recency', 'Frequency', 'Monetary']

# 5.3 Feature scaling for hierarchical clustering
scaler_hierarchical = StandardScaler()
df_hierarchical_scaled = scaler_hierarchical.fit_transform(df_hierarchical[features_hierarchical])

# 5.4 Plot Dendrogram to determine the number of clusters
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(df_hierarchical_scaled, method="ward"))
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# 5.5 Define number of clusters based on the dendrogram
n_clusters_hierarchical = 4 

# 5.6 Apply Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_hierarchical, metric="euclidean", linkage="ward")
df_hierarchical["Hierarchical_Cluster"] = agg_clustering.fit_predict(df_hierarchical_scaled)

# 5.7 Evaluate the clustering
silhouette_hierarchical = silhouette_score(df_hierarchical_scaled, df_hierarchical["Hierarchical_Cluster"])
davies_bouldin_hierarchical = davies_bouldin_score(df_hierarchical_scaled, df_hierarchical["Hierarchical_Cluster"])

# Display clustering metrics
print(f"Silhouette Score (Hierarchical): {silhouette_hierarchical:.4f}")
print(f"Davies-Bouldin Index (Hierarchical): {davies_bouldin_hierarchical:.4f}")

# 5.8 Analyze cluster distribution
numeric_columns = df_hierarchical.select_dtypes(include=[np.number]).columns
print("\nHierarchical Clustering - Cluster Summary:")
print(df_hierarchical.groupby("Hierarchical_Cluster")[numeric_columns].mean())

# 5.9 Visualize cluster characteristics
cluster_means_hierarchical = df_hierarchical.groupby("Hierarchical_Cluster")[features_hierarchical].mean()

plt.figure(figsize=(12, 6))
sns.heatmap(cluster_means_hierarchical, annot=True, cmap="coolwarm")
plt.title("Cluster Profiles (Hierarchical Clustering)")
plt.show()

# 5.10 Distribution of spending by cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x="Hierarchical_Cluster", y="Monetary", data=df_hierarchical)
plt.title("Monetary Spend Distribution Across Clusters (Hierarchical)")
plt.show()

# 5.11 Define personas based on cluster behaviors
hierarchical_personas = {
    0: "Premium Shopper (High spending and good frequency, indicating premium buyers.)",
    1: "Occasional Buyer (They spend less per transaction and have high recency, meaning they haven't purchased recently but may return occasionally.)",
    2: "At-Risk Customer (Low engagement, low monetary spend, and moderate frequency. At risk of churning soon.)",
    3: "High-Frequency Budget Shopper(Very high frequency, but moderate spending. They buy often but prioritize affordability over premium products.)"
}

# Map personas to clusters
df_hierarchical["Hierarchical_Persona"] = df_hierarchical["Hierarchical_Cluster"].map(hierarchical_personas)

# Summarize profiles with persona labels
cluster_profiles_hierarchical = df_hierarchical.groupby("Hierarchical_Cluster")[features_hierarchical].mean().reset_index()
cluster_profiles_hierarchical["Persona"] = cluster_profiles_hierarchical["Hierarchical_Cluster"].map(hierarchical_personas)

print("\nHierarchical Clustering - Customer Personas:")
print(cluster_profiles_hierarchical)

# -------------------------------------------
# Section 6: DBSCAN (Density-Based Clustering)
# -------------------------------------------

# 6.1 Create a separate copy for DBSCAN clustering
df_dbscan = df_base.copy()

# 6.2 Feature selection for DBSCAN
features_dbscan = ['Recency', 'Frequency', 'Monetary']

# 6.3 Feature scaling for DBSCAN using StandardScaler
scaler_dbscan = StandardScaler()
df_dbscan_scaled = scaler_dbscan.fit_transform(df_dbscan[features_dbscan])

# 6.4 K-distance graph to determine epsilon
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(df_dbscan_scaled)
distances, indices = neighbors_fit.kneighbors(df_dbscan_scaled)

# Sort and plot the distances
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.figure(figsize=(12, 6))
plt.plot(distances)
plt.title("K-Distance Graph")
plt.xlabel("Data Points")
plt.ylabel("Epsilon")
plt.show()

# 6.5 Fine-tuning around best found parameters 
eps_values = [0.58, 0.6, 0.62]  
min_samples_values = [5, 8, 10]  

dbscan_optimized_results = {}

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(df_dbscan_scaled)
        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        num_noise = list(clusters).count(-1)

        # Compute Silhouette Score (only if more than 1 cluster exists)
        if num_clusters > 1:
            silhouette = silhouette_score(df_dbscan_scaled, clusters)
        else:
            silhouette = None

        dbscan_optimized_results[(eps, min_samples)] = {
            "Number of Clusters": num_clusters,
            "Noise Points": num_noise,
            "Silhouette Score": silhouette
        }

# Display DBSCAN parameter tuning results
print("\nDBSCAN Parameter Tuning Results:")
for params, results in dbscan_optimized_results.items():
    print(f"Parameters: {params} - Clusters: {results['Number of Clusters']}, Noise Points: {results['Noise Points']}")

# 6.6 Apply DBSCAN with optimal parameters    
optimal_eps = 0.62
optimal_min_samples = 5
dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)

# Apply DBSCAN and add cluster labels
df_dbscan["DBSCAN_Cluster"] = dbscan.fit_predict(df_dbscan_scaled)

# 6.7 Count clusters and noise points (-1 indicates noise)
cluster_counts = df_dbscan["DBSCAN_Cluster"].value_counts()
outliers_count = cluster_counts[-1] if -1 in cluster_counts else 0

# Display results
print("\nDBSCAN Clustering Results:")
print("Cluster Distribution:\n", cluster_counts)
print("\nNumber of Outliers Detected:", outliers_count)

# 6.8 Evaluate DBSCAN clustering (excluding noise points)
filtered_df_dbscan = df_dbscan[df_dbscan["DBSCAN_Cluster"] != -1]
filtered_scaled_dbscan = df_dbscan_scaled[df_dbscan["DBSCAN_Cluster"] != -1]

# Compute silhouette score (if there are valid clusters)
if len(set(filtered_df_dbscan["DBSCAN_Cluster"])) > 1:
    silhouette_dbscan = silhouette_score(filtered_scaled_dbscan, filtered_df_dbscan["DBSCAN_Cluster"])
    print(f"\nSilhouette Score for DBSCAN: {silhouette_dbscan:.4f}")
else:
    silhouette_dbscan = None
    print("\nSilhouette Score not available (DBSCAN found only one cluster).")

# 6.9 Visualize DBSCAN clustering results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_dbscan["Recency"], y=df_dbscan["Monetary"], hue=df_dbscan["DBSCAN_Cluster"], palette="tab10")
plt.title("DBSCAN Clustering Results (Recency vs Monetary)")
plt.show()

# 6.10 Define personas for DBSCAN clusters
dbscan_personas = {
    -1: "Noise/Outliers (Unique customer behaviors that don't fit standard patterns)",
    0: "Premium Loyalist (High monetary value, moderate frequency, and high engagement.)",
    1: "High-Frequency Deal Seeker (This group buys very frequently but spends moderately, likely prioritizing discounts and promotions.)",
    2: "Low-Value Irregular Shopper (They have low spending, low frequency, and are not very engaged, meaning they shop infrequently and don't interact much.)"
}

# Map personas to clusters
df_dbscan["DBSCAN_Persona"] = df_dbscan["DBSCAN_Cluster"].map(dbscan_personas)

# Create a profile summary with persona labels
dbscan_profiles = df_dbscan.groupby("DBSCAN_Cluster")[features_dbscan].mean().reset_index()
dbscan_profiles["Persona"] = dbscan_profiles["DBSCAN_Cluster"].map(dbscan_personas)

print("\nDBSCAN Customer Personas:")
print(dbscan_profiles)

# -------------------------------------------
# Section 7: RFME Analysis (Recency, Frequency, Monetary, Engagement)
# -------------------------------------------

# 7.1 Create a separate copy for RFME analysis
df_rfme = df_base.copy()

# 7.2 Feature selection for RFME model
rfme_features = ["Recency", "Frequency", "Monetary", "Engagement_Score"]

# 7.3 Normalize the RFME features (min-max scaling is good for this analysis)
scaler_rfme = MinMaxScaler()
rfme_scaled_array = scaler_rfme.fit_transform(df_rfme[rfme_features])
df_rfme_scaled = pd.DataFrame(rfme_scaled_array, columns=rfme_features)

print("\nRFME Features (Normalized):")
print(df_rfme_scaled.head())

# 7.4 Apply K-means for RFME Clustering

# Use the Elbow Method to determine optimal K
inertia_rfme = []
for k in range(1, 11):
    kmeans_rfme = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_rfme.fit(df_rfme_scaled)
    inertia_rfme.append(kmeans_rfme.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia_rfme, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K (RFME)")
plt.show()

# 7.5 Apply optimal RFME clustering (6 clusters based on visualization)
optimal_k_rfme = 6
kmeans_rfme = KMeans(n_clusters=optimal_k_rfme, random_state=42, n_init=10)
df_rfme["RFME_Cluster"] = kmeans_rfme.fit_predict(df_rfme_scaled)

# 7.6 Evaluate RFME clustering
silhouette_rfme = silhouette_score(df_rfme_scaled, df_rfme["RFME_Cluster"])
davies_bouldin_rfme = davies_bouldin_score(df_rfme_scaled, df_rfme["RFME_Cluster"])

print("\nRFME Clustering Evaluation:")
print(f"Silhouette Score: {silhouette_rfme:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_rfme:.4f}")

# 7.7 Visualize RFME Cluster Profiles
cluster_means_rfme = df_rfme.groupby("RFME_Cluster")[rfme_features].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means_rfme, annot=True, cmap="viridis")
plt.title("RFME Cluster Profiles")
plt.show()

# 7.8 Define and Interpret RFME Cluster Personas
rfme_personas = {
    0: "High-Value Loyal Customer (High monetary spend and engagement, moderate frequency, and moderate recency. They are high-value customers who consistently engage with the brand.)",
    1: "Recent Engaged Shopper (High engagement score and low recency, meaning they are actively shopping but not frequently. They may become more loyal over time.)",
    2: "Frequent Low-Spender (Extremely high frequency but low monetary spend. They buy very often but in small amounts.)",
    3: "Infrequent Low-Spender (Low monetary and moderate engagement, meaning they buy occasionally but don't spend much.)",
    4: "Dormant High-Value Customer (High recency (meaning they haven't purchased in a while), but decent frequency and spending when they were active. They may be worth re-engaging.)",
    5: "High-Frequency Premium Buyer (High frequency and monetary spend, meaning they shop often and spend well, but their engagement score is slightly lower.)"
}

# Map personas to clusters
df_rfme["RFME_Persona"] = df_rfme["RFME_Cluster"].map(rfme_personas)

# Display the cluster profiles with persona descriptions
cluster_summary_rfme = df_rfme.groupby("RFME_Cluster")[rfme_features].mean().reset_index()
cluster_summary_rfme["Persona"] = cluster_summary_rfme["RFME_Cluster"].map(rfme_personas)

print("\nRFME Cluster Summary:")
print(cluster_summary_rfme)

# 7.9 Visualize RFME Segments
plt.figure(figsize=(12, 5))
sns.scatterplot(x="Recency", y="Monetary", hue="RFME_Persona", data=df_rfme, palette="tab10")
plt.title("RFME Customer Segments")
plt.show()

# -------------------------------------------
# Section 8: Gaussian Mixture Models (GMM)
# -------------------------------------------

# 8.1 Create a separate copy for GMM analysis
df_gmm = df_base.copy()

# 8.2 Feature selection and scaling for GMM
features_gmm = ['Recency', 'Frequency', 'Monetary', 'Engagement_Score']
scaler_gmm = StandardScaler()
df_gmm_scaled = scaler_gmm.fit_transform(df_gmm[features_gmm])

# 8.3 Perform PCA to reduce dimensionality before GMM
pca_gmm = PCA(n_components=2)
X_pca_gmm = pca_gmm.fit_transform(df_gmm_scaled)

# 8.4 Find optimal number of components for GMM using BIC
n_components_range = range(2, 10)
bic_scores = []
aic_scores = []
gmm_models = {}

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_pca_gmm)
    bic_scores.append(gmm.bic(X_pca_gmm))
    aic_scores.append(gmm.aic(X_pca_gmm))
    gmm_models[n_components] = gmm

# Find optimal number of components (lowest BIC score)
optimal_n_components = n_components_range[np.argmin(bic_scores)]
print(f"\nOptimal Number of GMM Components (using BIC): {optimal_n_components}")

# 8.5 Try alternative numbers of components for comparison
alternative_components = [5]  # Testing 5 components based on results from other models
gmm_alternative_results = {}

for n_components in alternative_components:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_pca_gmm)
    labels = gmm.predict(X_pca_gmm)
    
    # Calculate silhouette score
    silhouette = silhouette_score(X_pca_gmm, labels)
    gmm_alternative_results[n_components] = {"Silhouette Score": silhouette}

# Display alternative results
for n_components, results in gmm_alternative_results.items():
    print(f"GMM with {n_components} Components - Silhouette Score: {results['Silhouette Score']:.4f}")

# 8.6 Apply GMM with 5 components (more interpretable)
gmm_final = GaussianMixture(n_components=5, random_state=42)
gmm_final.fit(X_pca_gmm)
df_gmm["GMM_Cluster"] = gmm_final.predict(X_pca_gmm)

# Calculate silhouette score
silhouette_gmm = silhouette_score(X_pca_gmm, df_gmm["GMM_Cluster"])
print(f"GMM Final Silhouette Score: {silhouette_gmm:.4f}")

# 8.7 Visualize GMM Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca_gmm[:, 0], y=X_pca_gmm[:, 1], hue=df_gmm["GMM_Cluster"], palette="viridis")
plt.title("GMM Clustering Results (PCA)")
plt.show()

# 8.8 Define GMM personas based on cluster characteristics
gmm_profiles = df_gmm.groupby("GMM_Cluster")[features_gmm].mean()
print("\nGMM Cluster Profiles:")
print(gmm_profiles)

gmm_personas = {
    0: "Low Engagement Browser (Low across all metrics, particularly engagement score. These customers browse occasionally but rarely convert to significant purchases.)",
    1: "Moderate Value Loyal (Medium recency, frequency, and monetary values with good engagement. These are solid middle-tier customers.)",
    2: "High-Value Premium (High monetary value with good recency and frequency. These are premium customers who make regular valuable purchases.)",
    3: "Recent Low-Value Explorer (Low recency (recent purchases) but low monetary value. These may be new customers exploring the brand with small initial purchases.)",
    4: "Occasional Big Spender (High monetary value but low frequency. These customers shop infrequently but spend significantly when they do.)"
}

# Map personas to clusters
df_gmm["GMM_Persona"] = df_gmm["GMM_Cluster"].map(gmm_personas)

# Display summary with persona descriptions
gmm_summary = df_gmm.groupby("GMM_Cluster")[features_gmm].mean().reset_index()
gmm_summary["Persona"] = gmm_summary["GMM_Cluster"].map(gmm_personas)
print("\nGMM Persona Summary:")
print(gmm_summary)

# -------------------------------------------
# Section 9: K-Prototypes for Mixed Data Types
# -------------------------------------------

try:
    # 9.1 Create a separate copy for K-Prototypes analysis
    df_kproto = df_base.copy()

    # 9.2 Select features for K-Prototypes
    categorical_features = ['Gender', 'Category', 'Size', 'Color', 'Season', 'Subscription Status', 'Payment Method']
    numerical_features = ['Age', 'Frequency', 'Monetary', 'Engagement_Score', 'Recency']

    # 9.3 Ensure categorical features are strings
    df_kproto[categorical_features] = df_kproto[categorical_features].astype(str)

    # 9.4 Prepare data for K-Prototypes
    X_kproto = df_kproto[numerical_features + categorical_features].values
    categorical_indices = [len(numerical_features) + i for i in range(len(categorical_features))]

    # 9.5 Apply K-Prototypes clustering
    kproto = KPrototypes(n_clusters=4, init='Huang', random_state=42)
    clusters = kproto.fit_predict(X_kproto, categorical=categorical_indices)
    df_kproto['KPrototypes_Cluster'] = clusters

    # 9.6 Analyze KPrototypes clusters
    print("\nK-Prototypes Clustering Results:")
    print(df_kproto[["Customer ID", "KPrototypes_Cluster"]].head())

    # 9.7 Compute mean values for numerical features per cluster
    cluster_means_kproto = df_kproto.groupby("KPrototypes_Cluster")[numerical_features].mean()

    # 9.8 Visualize cluster profiles
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means_kproto, annot=True, cmap="coolwarm")
    plt.title("K-Prototypes Cluster Profiles (Numerical Features)")
    plt.show()

    # 9.9 Get mode (most frequent category) for categorical features in each cluster
    cluster_modes_kproto = df_kproto.groupby("KPrototypes_Cluster")[categorical_features].agg(lambda x: x.mode()[0])
    print("\nMost Common Categorical Values per Cluster:")
    print(cluster_modes_kproto)

    # 9.10 Define K-Prototypes personas
    kproto_personas = {
        0: "Luxury High-Spender (High frequency and moderate monetary spend, suggesting they are engaged, frequent shoppers who make regular purchases.)",
        1: "Infrequent High-Spender (Older age with moderate spending but low frequency, indicating they buy less often but in higher amounts.)",
        2: "Loyal Subscription Shopper (Moderate frequency, moderate spend, and steady engagement suggest customers who consistently purchase—likely through subscriptions or repeat orders.)",
        3: "Moderate-Frequency Engaged Shopper (Lower recency and decent engagement, meaning they may shop semi-regularly but without strong spending patterns.)"
    }

    # Map personas to clusters
    df_kproto["KPrototypes_Persona"] = df_kproto["KPrototypes_Cluster"].map(kproto_personas)

    # Display persona summary
    cluster_summary_kproto = df_kproto.groupby("KPrototypes_Cluster")[numerical_features].mean().reset_index()
    cluster_summary_kproto["Persona"] = cluster_summary_kproto["KPrototypes_Cluster"].map(kproto_personas)
    print("\nK-Prototypes Cluster Summary:")
    print(cluster_summary_kproto)

    # 9.11 Calculate silhouette score for K-Prototypes
    # Convert categorical columns to numerical for silhouette score calculation
    df_kproto_num = df_kproto.copy()
    df_kproto_num[categorical_features] = df_kproto_num[categorical_features].astype('category').apply(lambda x: x.cat.codes)
    X_kproto_num = df_kproto_num[numerical_features + categorical_features].values
    silhouette_kproto = silhouette_score(X_kproto_num, df_kproto["KPrototypes_
