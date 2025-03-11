import os
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')

# 🎭 Define Customer Personas with Updated GMM Insights and Custom Icons
persona_details = {
    "High-Value Loyal Customer": {
        "img": "https://cdn-icons-png.flaticon.com/512/1379/1379505.png",
        "size": "899 customers (23.6% of total customers)",
        "recency": 37.87,
        "frequency": 11.61,
        "monetary": 0.75,
        "engagement_score": 4.33,
        "purchase_avg": "$75.00",
        "conversion_rate": "8.5%",
        "revenue_increase": "$180K",
        "purchase_frequency": "Monthly (75%), Quarterly (15%)",
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
        "img": "https://cdn-icons-png.flaticon.com/512/1260/1260235.png",
        "size": "990 customers (26.0% of total customers)",
        "recency": 11.69,
        "frequency": 11.32,
        "monetary": 0.52,
        "engagement_score": 4.43,
        "purchase_avg": "$52.46",
        "conversion_rate": "6.2%",
        "revenue_increase": "$140K",
        "purchase_frequency": "Weekly (60%), Bi-Weekly (25%)",
        "previous_purchases": "11.3",
        "avg_rating": "4.4/5",
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
    "High-Frequency Premium Buyer": {
        "img": "https://cdn-icons-png.flaticon.com/512/4128/4128176.png",
        "size": "200 customers (5.2% of total customers)",  # Estimated from the data
        "recency": 24.38,
        "frequency": 12.10,
        "monetary": 0.76,
        "engagement_score": 3.10,
        "purchase_avg": "$76.00",
        "conversion_rate": "7.8%",
        "revenue_increase": "$160K",
        "purchase_frequency": "Weekly (90%), Bi-Weekly (8%)",
        "previous_purchases": "24.5",
        "avg_rating": "4.7/5",
        "discount_usage": "25.3%",
        "promo_code_usage": "30.7%",
        "top_categories": ["Premium Clothing", "Luxury Accessories"],
        "marketing_recommendations": [
            "Create exclusive VIP early access to new collections",
            "Develop personalized concierge shopping service",
            "Offer complimentary styling consultations",
            "Implement premium loyalty program with exclusive benefits",
            "Host invitation-only events and product showcases"
        ],
        "product_strategy": "Focus on high-end product lines, limited editions, and personalized shopping experiences."
    },
    "Dormant High-Value Customer": {
        "img": "https://cdn-icons-png.flaticon.com/512/8633/8633496.png",
        "size": "568 customers (14.9% of total customers)",
        "recency": 38.31,
        "frequency": 11.50,
        "monetary": 0.22,
        "engagement_score": 3.72,
        "purchase_avg": "$65.40",
        "conversion_rate": "4.2%",
        "revenue_increase": "$120K",
        "purchase_frequency": "Annually (60%), Semi-Annually (30%)",
        "previous_purchases": "22.3",
        "avg_rating": "3.9/5",
        "discount_usage": "40.7%",
        "promo_code_usage": "40.7%",
        "top_categories": ["Clothing", "Home Goods"],
        "marketing_recommendations": [
            "Create targeted re-engagement campaigns with personalized offers",
            "Implement win-back strategies with exclusive discounts",
            "Send personalized product updates based on past purchases",
            "Develop automated email sequences highlighting new products",
            "Use retargeting ads with special comeback incentives"
        ],
        "product_strategy": "Showcase new collections and complementary products to past purchases."
    },
    "Frequent Low-Spender": {
        "img": "https://cdn-icons-png.flaticon.com/512/3135/3135823.png",
        "size": "424 customers (11.1% of total customers)",
        "recency": 25.77,
        "frequency": 52.00,
        "monetary": 0.49,
        "engagement_score": 3.76,
        "purchase_avg": "$25.45",
        "conversion_rate": "5.5%",
        "revenue_increase": "$80K",
        "purchase_frequency": "Weekly (70%), Bi-Weekly (25%)",
        "previous_purchases": "52.0",
        "avg_rating": "3.8/5",
        "discount_usage": "65.9%",
        "promo_code_usage": "72.9%",
        "top_categories": ["Accessories", "Small Items"],
        "marketing_recommendations": [
            "Bundle frequently purchased items for value pricing",
            "Create subscription options for regularly purchased products",
            "Implement tiered discount structure to encourage larger basket sizes",
            "Develop loyalty program focusing on purchase frequency",
            "Use targeted upselling strategies for complementary products"
        ],
        "product_strategy": "Focus on affordable product lines with opportunities for upselling and cross-selling."
    },
    "Infrequent Low-Spender": {
        "img": "https://cdn-icons-png.flaticon.com/512/18332/18332021.png",
        "size": "1019 customers (26.7% of total customers)",
        "recency": 13.55,
        "frequency": 11.46,
        "monetary": 0.24,
        "engagement_score": 3.22,
        "purchase_avg": "$24.00",
        "conversion_rate": "3.2%",
        "revenue_increase": "$60K",
        "purchase_frequency": "Annually (80%), Semi-Annually (15%)",
        "previous_purchases": "11.5",
        "avg_rating": "3.2/5",
        "discount_usage": "55.3%",
        "promo_code_usage": "60.3%",
        "top_categories": ["Sale Items", "Seasonal Products"],
        "marketing_recommendations": [
            "Create special offers focused on value and affordability",
            "Implement automated engagement campaigns to increase purchase frequency",
            "Develop targeted promotions for seasonal shopping events",
            "Use strategic discounting to encourage larger basket sizes",
            "Create personalized recommendations based on past browsing behavior"
        ],
        "product_strategy": "Highlight budget-friendly options and entry-level products with clear value propositions."
    }
}

# Create a DataFrame with RFM metrics from Image 1
rfm_data = {
    "RFME_Persona": ["Dormant High-Value Customer", "Frequent Low-Spender", 
                     "High-Frequency Premium Buyer", "High-Value Loyal Customer", 
                     "Infrequent Low-Spender", "Recent Engaged Shopper"],
    "Recency": [38.31, 25.77, 24.38, 37.87, 13.55, 11.69],
    "Frequency": [11.50, 52.00, 12.10, 11.61, 11.46, 11.32],
    "Monetary": [0.22, 0.49, 0.76, 0.75, 0.24, 0.52],
    "Engagement_Score": [3.72, 3.76, 3.10, 4.33, 3.22, 4.43]
}

rfm_df = pd.DataFrame(rfm_data)

# Create a DataFrame with GMM persona counts from Image 2
gmm_data = {
    "GMM_Persona": ["Infrequent Low-Spender", "Recent Engaged Shopper", 
                    "High-Value Loyal Customer", "Dormant High-Value Customer", 
                    "Frequent Low-Spender"],
    "Count": [1019, 990, 899, 568, 424]
}

gmm_df = pd.DataFrame(gmm_data)

# 🎯 Streamlit UI Setup
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("📊 Customer Segmentation Analysis")
st.write("Click on a customer segment to explore its characteristics, marketing recommendations, and product strategy.")

# 📌 Display clickable persona images above names in a row
cols = st.columns(len(persona_details))
selected_persona = None

for i, (persona_name, persona) in enumerate(persona_details.items()):
    with cols[i]:
        st.image(persona["img"], width=100)
        if st.button(persona_name):
            selected_persona = persona_name

# Add overall metrics
st.write("### 📈 Overall Customer Segments")
col1, col2 = st.columns(2)

with col1:
    st.write("#### GMM Segmentation Counts")
    fig_gmm = px.bar(gmm_df, x="GMM_Persona", y="Count", 
                    title="Customer Segments Distribution", 
                    color="GMM_Persona",
                    labels={"GMM_Persona": "Customer Segment", "Count": "Number of Customers"})
    st.plotly_chart(fig_gmm, use_container_width=True)

with col2:
    st.write("#### RFM Metrics by Segment")
    rfm_metrics = rfm_df.melt(id_vars=["RFME_Persona"], 
                             value_vars=["Recency", "Frequency", "Monetary", "Engagement_Score"],
                             var_name="Metric", value_name="Value")
    fig_rfm = px.bar(rfm_metrics, x="RFME_Persona", y="Value", color="Metric",
                    barmode="group", title="RFM Metrics by Customer Segment",
                    labels={"RFME_Persona": "Customer Segment", "Value": "Score"})
    st.plotly_chart(fig_rfm, use_container_width=True)

# 📌 Show details when a persona is selected
if selected_persona:
    persona = persona_details[selected_persona]
    st.markdown(f"## 🎭 {selected_persona}", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image(persona["img"], width=150)
        st.write(f"**Size:** {persona['size']}")
        st.write(f"**Average Purchase:** {persona['purchase_avg']}")
        st.write(f"**Conversion Rate:** {persona['conversion_rate']}")
    
    with col2:
        # RFM Metrics
        rfm_cols = st.columns(4)
        with rfm_cols[0]:
            st.metric("Recency", f"{persona['recency']:.2f}")
        with rfm_cols[1]:
            st.metric("Frequency", f"{persona['frequency']:.2f}")
        with rfm_cols[2]:
            st.metric("Monetary", f"{persona['monetary']:.2f}")
        with rfm_cols[3]:
            st.metric("Engagement", f"{persona['engagement_score']:.2f}")
            
        # Additional metrics
        st.write(f"**Expected Revenue Increase:** {persona['revenue_increase']}")
        st.write(f"**Purchase Frequency:** {persona['purchase_frequency']}")
        st.write(f"**Previous Purchases:** {persona['previous_purchases']}")
        st.write(f"**Average Rating:** {persona['avg_rating']}")
        st.write(f"**Discount Usage:** {persona['discount_usage']}")
        st.write(f"**Promo Code Usage:** {persona['promo_code_usage']}")
        st.write(f"**Top Categories:** {', '.join(persona['top_categories'])}")
    
    # Recommendations and strategy
    st.write("### 📌 Marketing Recommendations")
    for rec in persona["marketing_recommendations"]:
        st.write(f"- {rec}")
    
    st.write(f"### 🎯 Product Strategy")
    st.write(persona['product_strategy'])
    
    # 📊 Segment comparison with other segments
    st.write("### 📊 Segment Comparison")
    
    # Create comparison dataframe
    segment_df = pd.DataFrame({
        "Metric": ["Recency", "Frequency", "Monetary", "Engagement"],
        selected_persona: [
            persona['recency'],
            persona['frequency'],
            persona['monetary'],
            persona['engagement_score']
        ]
    })
    
    # Add other segments for comparison
    for other_persona, other_data in persona_details.items():
        if other_persona != selected_persona:
            segment_df[other_persona] = [
                other_data['recency'],
                other_data['frequency'],
                other_data['monetary'],
                other_data['engagement_score']
            ]
    
    comparison_df = segment_df.melt(id_vars=["Metric"], var_name="Segment", value_name="Value")
    fig = px.bar(comparison_df, x="Metric", y="Value", color="Segment", barmode="group",
                title=f"Comparison of {selected_persona} with Other Segments",
                labels={"Value": "Score", "Metric": "RFM Metrics"})
    st.plotly_chart(fig)

# Business Impact section
st.write("### 📌 Business Impact")
st.write("- Personalized marketing campaigns based on segments can enhance conversions.")
st.write("- GMM segmentation provides data-driven customer targeting.")
st.write("- Strategic focus on high-value segments is projected to increase revenue by $580K annually.")
st.write("- Reactivating dormant high-value customers could yield a $120K revenue increase.")
st.write("- Converting Infrequent Low-Spenders to Frequent Low-Spenders would increase purchase frequency by 40%.")
