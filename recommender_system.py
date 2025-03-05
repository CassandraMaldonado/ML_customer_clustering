import streamlit as st
import pandas as pd
import numpy as np

# Title and description for the Streamlit app
st.title("Item-Based Collaborative Filtering Recommendation System")
st.write("This app recommends products to users based on their past purchase behavior using an item-based collaborative filtering approach.")

# Function to load data (with caching to avoid reloading on each interaction)
@st.cache_data
def load_data():
    # Load the dataset of shopping behavior.
    # Assumes the CSV file is in the same directory as this script or provide the correct path.
    return pd.read_csv("shopping_behavior_updated.csv")

# Function to compute item similarity matrix (with caching for efficiency)
@st.cache_data
def compute_item_similarity(df):
    """
    Compute item-to-item cosine similarity based on user purchase history.
    Returns:
        similarity_matrix (numpy.ndarray): 2D array of item cosine similarity scores.
        item_to_index (dict): Mapping from item name to index in the similarity matrix.
        index_to_item (dict): Mapping from index in the similarity matrix to item name.
        user_item_matrix (pd.DataFrame): Binary user-item matrix (rows=users, cols=items).
    """
    # Create a user-item matrix of purchase counts (or 1 if purchased, 0 if not)
    user_item = df.groupby(['Customer ID', 'Item Purchased']).size().unstack(fill_value=0)
    # Convert counts to binary values: 1 if user purchased the item, 0 otherwise
    user_item_binary = (user_item > 0).astype(int)
    
    # Compute co-occurrence matrix (item-item): how many users purchased each pair of items
    X = user_item_binary.values  # shape: [num_users, num_items]
    cooccurrence = X.T.dot(X)    # shape: [num_items, num_items], cooccurrence[i,j] = number of users who bought both item i and item j
    
    # Compute cosine similarity matrix from co-occurrence
    # cosine_sim(i,j) = cooccurrence[i,j] / sqrt(cooccurrence[i,i] * cooccurrence[j,j])
    item_counts = np.diag(cooccurrence).astype(float)  # number of purchasers for each item (diagonal of cooccurrence)
    norm = np.outer(np.sqrt(item_counts), np.sqrt(item_counts))  # denominator matrix for normalization
    # Use numpy divide with where to avoid division by zero (in case an item has no purchases)
    similarity_matrix = np.divide(cooccurrence, norm, out=np.zeros_like(cooccurrence, dtype=float), where=norm != 0)
    # Set diagonal to 0 (an item is not considered similar to itself for recommendation purposes)
    np.fill_diagonal(similarity_matrix, 0)
    
    # Create mappings for item indices and names for easy lookup
    items = list(user_item_binary.columns)
    item_to_index = {item: idx for idx, item in enumerate(items)}
    index_to_item = {idx: item for item, idx in item_to_index.items()}
    
    return similarity_matrix, item_to_index, index_to_item, user_item_binary

# Load the data and precompute item similarities
data = load_data()
similarity_matrix, item_to_index, index_to_item, user_item_matrix = compute_item_similarity(data)

# Streamlit UI: User selection for which we want to get recommendations
user_id = st.number_input(
    "Select User ID for recommendations",
    min_value=int(data["Customer ID"].min()),
    max_value=int(data["Customer ID"].max()),
    value=int(data["Customer ID"].min()),
    step=1
)

# Check if the selected user exists in our dataset
if user_id not in user_item_matrix.index:
    st.error(f"User ID {user_id} not found in the dataset. Please select a valid User ID.")
else:
    # Retrieve the items that the user has purchased
    user_vector = user_item_matrix.loc[user_id]
    purchased_items = user_vector[user_vector > 0].index.tolist()
    
    # Display the user's purchase history
    st.markdown(f"**User {user_id} Purchase History:** " +
                (", ".join(purchased_items) if purchased_items else "_(No purchases found.)_"))
    
    if purchased_items:
        # Calculate recommendation scores for each item (sum of similarities with all items the user purchased)
        item_scores = np.zeros(similarity_matrix.shape[0])
        for item in purchased_items:
            item_idx = item_to_index[item]
            item_scores += similarity_matrix[item_idx]
        # Exclude items already purchased by setting their scores to a negative number
        for item in purchased_items:
            item_scores[item_to_index[item]] = -1
        
        # Select the top 5 highest scoring items as recommendations
        top_indices = item_scores.argsort()[::-1][:5]
        recommended_items = [index_to_item[idx] for idx in top_indices if item_scores[idx] > 0]
        
        # If no recommendations are found (e.g., no overlapping purchase history), fall back to top popular items
        if not recommended_items:
            item_popularity = user_item_matrix.sum(axis=0)  # total purchases of each item
            popular_items = item_popularity.sort_values(ascending=False).index.tolist()
            recommended_items = [item for item in popular_items if item not in purchased_items][:5]
        
        # Display the recommended items list
        st.subheader("Recommended Items:")
        if recommended_items:
            for i, rec_item in enumerate(recommended_items, start=1):
                st.write(f"{i}. **{rec_item}**")
        else:
            st.write("*(No recommendations available)*")
    else:
        st.write("Since this user has no purchase history, recommendations cannot be generated.")
