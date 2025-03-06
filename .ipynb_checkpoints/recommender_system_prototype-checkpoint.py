import streamlit as st
import pandas as pd
import numpy as np

def load_data():
    """Load the static dataset."""
    file_path = "output_w_prod_cluster.csv"  # Static file
    return pd.read_csv(file_path)

def get_top_items_by_season(df, top_n=3):
    """Get the most popular items per season, excluding the purchased item later."""
    return df.groupby("Season")["Item Purchased"].apply(lambda x: x.value_counts().index[:top_n]).to_dict()

def get_top_colors_by_season(df, top_n=3):
    """Get the most common colors per season."""
    return df.groupby("Season")["Color"].apply(lambda x: x.value_counts().index[:top_n]).to_dict()

def recommend_items_with_colors(season, purchased_item, purchased_color, top_items, top_colors):
    """Recommend the top three most popular items for the season (excluding purchased) with color preferences."""
    season_items = top_items.get(season, [])
    season_items = [item for item in season_items if item != purchased_item]
    recommended_items = season_items[:3] if len(season_items) >= 3 else season_items
    
    season_colors = top_colors.get(season, [])
    recommended_colors = season_colors[:3] if len(season_colors) >= 3 else season_colors
    
    return list(zip(recommended_items, recommended_colors)) if recommended_items else [("No recommendation", "No color")]

# Streamlit UI
st.set_page_config(page_title="Product Recommender", layout="wide")
st.title("🛍️ Product Recommendation System")

# Load data
st.sidebar.header("Dataset Information")
df = load_data()

top_items_by_season = get_top_items_by_season(df)  # Ensure it's defined globally
top_colors_by_season = get_top_colors_by_season(df)

if df is not None:
    st.sidebar.success("Dataset Loaded Successfully!")
    
    # User input for Customer ID
    customer_id = st.sidebar.text_input("Enter Customer ID:")
    
    if customer_id:
        # Check if Customer ID exists
        if customer_id in df.index.astype(str):
            customer_row = df.loc[df.index.astype(str) == customer_id].iloc[0]
            
            st.subheader(f"Customer {customer_id}'s Purchase Information")
            st.write(f"**Item Purchased:** {customer_row['Item Purchased']}")
            st.write(f"**Color:** {customer_row['Color']}")
            st.write(f"**Season:** {customer_row['Season']}")
            
            # Generate recommendations for the customer
            recommendations = recommend_items_with_colors(
                customer_row["Season"], customer_row["Item Purchased"], customer_row["Color"], top_items_by_season, top_colors_by_season
            )
            
            st.subheader("📌 Recommended Products")
            for i, (item, color) in enumerate(recommendations, 1):
                st.write(f"**Recommendation {i}:** {item} in {color}")
            
        else:
            st.error("Customer ID not found in the dataset.")
    
    st.success("✅ Ready to generate recommendations!")