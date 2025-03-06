import os

file_path = "Models_Final.ipynb"
# Convert the Jupyter Notebook to a Python script
os.system(f"jupyter nbconvert --to script {file_path}")

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')

import streamlit as st
import plotly.express as px
import pandas as pd

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')

# 🎭 Define Customer Personas with Updated GMM Insights and Icons
persona_details = {
    "High-Value Loyal Customer": {
        "img": "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        "size": "405 customers (10.4% of total customers)",
        "purchase_avg": "$39.65",
        "purchase_frequency": "Fortnightly (75), Every 3 Months (72)",
        "previous_purchases": "35.9",
        "avg_rating": "4.3/5",
        "discount_usage": "43.2%",
        "promo_code_usage": "43.2%",
        "top_categories": ["Clothing", "Accessories"],
        "marketing_recommendations": [
            "Develop a rewards program with tiered benefits to enhance loyalty",
            "Offer personalized recommendations based on past purchases",
            "Provide special early-access to new product launches",
            "Encourage referrals with discounts or perks",
            "Send engagement-driven content such as styling tips, usage guides, etc."
        ],
        "product_strategy": "Offer personalized product bundles and exclusive early access to promotions."
    },
    "Recent Engaged Shopper": {
        "img": "https://cdn-icons-png.flaticon.com/512/3135/3135762.png",
        "size": "429 customers (11.0% of total customers)",
        "purchase_avg": "$78.36",
        "purchase_frequency": "Bi-Weekly (83), Every 3 Months (76)",
        "previous_purchases": "13.5",
        "avg_rating": "3.2/5",
        "discount_usage": "41.7%",
        "promo_code_usage": "41.7%",
        "top_categories": ["Clothing", "Accessories"],
        "marketing_recommendations": [
            "Implement a nurturing email series introducing product benefits",
            "Encourage repeat purchases with 'next purchase' incentives",
            "Use social proof (reviews, testimonials) to build trust",
            "Showcase best-selling items in follow-up marketing campaigns",
            "Offer personalized first-purchase discounts to increase retention"
        ],
        "product_strategy": "Promote starter sets, introductory offers, and educational product content."
    },
    "Frequent Low-Spender": {
        "img": "https://cdn-icons-png.flaticon.com/512/3135/3135823.png",
        "size": "434 customers (11.1% of total customers)",
        "purchase_avg": "$41.45",
        "purchase_frequency": "Annually (88), Quarterly (79)",
        "previous_purchases": "16.0",
        "avg_rating": "3.2/5",
        "discount_usage": "42.9%",
        "promo_code_usage": "42.9%",
        "top_categories": ["Clothing", "Accessories"],
        "marketing_recommendations": [
            "Test various engagement strategies to refine segmentation",
            "Develop targeted outreach campaigns to drive conversions",
            "Analyze buying behavior for deeper insights into shopping patterns",
            "Implement automated re-engagement emails and promotions",
            "Create A/B tests for personalized product recommendations"
        ],
        "product_strategy": "Refine product recommendations based on customer behavior and purchase history."
    },
}

# 🎯 Streamlit UI Setup
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("📊 Customer Segmentation Analysis")
st.write("Click on a customer segment to explore its characteristics, marketing recommendations, and product strategy.")

# 📌 Display clickable persona images in a row
cols = st.columns(len(persona_details))
selected_persona = None

for i, (persona_name, persona) in enumerate(persona_details.items()):
    if cols[i].image(persona["img"], width=100):
        selected_persona = persona_name

# 📌 Show details when a persona is selected
if selected_persona:
    persona = persona_details[selected_persona]
    st.subheader(selected_persona)
    st.write(f"**Size:** {persona['size']}")
    st.write(f"**Average Purchase:** {persona['purchase_avg']}")
    st.write(f"**Purchase Frequency:** {persona['purchase_frequency']}")
    st.write(f"**Previous Purchases:** {persona['previous_purchases']}")
    st.write(f"**Average Rating:** {persona['avg_rating']}")
    st.write(f"**Discount Usage:** {persona['discount_usage']}")
    st.write(f"**Promo Code Usage:** {persona['promo_code_usage']}")
    st.write(f"**Top Categories:** {', '.join(persona['top_categories'])}")
    st.write("### 📌 Marketing Recommendations")
    st.write("\n".join([f"- {rec}" for rec in persona["marketing_recommendations"]]))
    st.write(f"### 🎯 Product Strategy: {persona['product_strategy']}")

    # 📊 Generate Graphs for the Selected Persona
    metrics = {
        "Recency": [30, 20, 25], 
        "Frequency": [10, 15, 8], 
        "Monetary": [80, 60, 50], 
        "Engagement_Score": [5, 10, 7]
    }
    metric_df = pd.DataFrame(metrics, index=["High-Value Loyal", "Recent Engaged", "Frequent Low-Spender"])
    
    st.write("### 📊 Customer Segment Metrics")
    fig = px.bar(metric_df.loc[selected_persona], title=f"Key Metrics for {selected_persona}", labels={"index": "Metrics", "value": "Mean Value"})
    st.plotly_chart(fig)

st.write("### 📌 Business Impact")
st.write("- Personalized marketing campaigns based on segments can enhance conversions.")
st.write("- GMM segmentation provides **data-driven** customer targeting.")
st.write("- By implementing targeted strategies, Acme Inc. aims to increase revenue by **$500K** annually.")
"""