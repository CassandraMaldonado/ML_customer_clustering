import streamlit as st
import pandas as pd
import numpy as np

st.title("Item-Based Collaborative Filtering (Item + Color)")
st.write("This app recommends products based on past purchases, considering item and color.")

# Function to load precomputed similarity matrix
@st.cache_data
def load_precomputed_data():
    similarity_df = pd.read_csv("precomputed_similarity.csv", index_col=0)
    user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0)
    
    return similarity_df, user_item_matrix

# Load precomputed data
similarity_df, user_item_matrix = load_precomputed_data()
items = similarity_df.index.tolist()

# Streamlit UI: Select User ID
user_id = st.number_input(
    "Select User ID for recommendations",
    min_value=int(user_item_matrix.index.astype(int).min()),
    max_value=int(user_item_matrix.index.astype(int).max()),
    value=int(user_item_matrix.index.astype(int).min()),
    step=1
)

if str(user_id) not in user_item_matrix.index:
    st.error(f"User ID {user_id} not found. Please select a valid User ID.")
else:
    # Retrieve purchased items
    user_vector = user_item_matrix.loc[str(user_id)]
    purchased_items = user_vector[user_vector > 0].index.tolist()

    st.markdown(f"**User {user_id} Purchase History:**")
    if purchased_items:
        for item in purchased_items:
            st.write(f"- {item}")
    else:
        st.write("_No purchases found._")

    # Generate Recommendations
    if purchased_items:
        item_scores = np.zeros(len(items))
        for item in purchased_items:
            if item in similarity_df.index:
                item_scores += similarity_df.loc[item].values
        
        for item in purchased_items:
            item_scores[items.index(item)] = -1  # Avoid recommending already purchased items

        top_indices = item_scores.argsort()[::-1][:5]
        recommended_items = [items[idx] for idx in top_indices if item_scores[idx] > 0]
        
        st.subheader("Recommended Items:")
        if recommended_items:
            for rec_item in recommended_items:
                st.write(f"- {rec_item}")
        else:
            st.write("_No recommendations available._")
