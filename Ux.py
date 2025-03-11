import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure required dependencies are installed
os.system('pip install streamlit pandas plotly matplotlib')

# 🎭 Define Customer Personas with Updated GMM Insights and Custom Icons
persona_details = {
    "size": "899 customers",
    "img": "https://cdn-icons-png.flaticon.com/512/1379/1379505.png",
    "size": "405 customers (10.4% of total customers)",
    "purchase_avg": "$39.65",
    "conversion_rate": "8.5%",
    "revenue_increase": "$150K",
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

    "size": "990 customers",
    "img": "https://cdn-icons-png.flaticon.com/512/1260/1260235.png",
    "size": "429 customers (11.0% of total customers)",
    "purchase_avg": "$78.36",
    "conversion_rate": "6.2%",
    "revenue_increase": "$120K",
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

    "size": "424 customers",
    "img": "https://cdn-icons-png.flaticon.com/512/3135/3135823.png",
    "size": "434 customers (11.1% of total customers)",
    "purchase_avg": "$41.45",
    "conversion_rate": "4.5%",
    "revenue_increase": "$80K",
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

    "size": "568 customers",
    "img": "https://cdn-icons-png.flaticon.com/512/8633/8633496.png",
    "size": "457 customers (11.7% of total customers)",
    "purchase_avg": "$40.60",
    "conversion_rate": "5.0%",
    "revenue_increase": "$90K",
    "purchase_frequency": "Every 3 Months (88), Fortnightly (81)",
    "previous_purchases": "14.3",
    "avg_rating": "4.3/5",
    "discount_usage": "40.7%",
    "promo_code_usage": "40.7%",
    "top_categories": ["Clothing", "Accessories"],
    "marketing_recommendations": [
    "Create an exclusive VIP membership program with premium perks",
    "Provide white-glove customer service and personal shopping assistance",
    "Offer invitation-only events and product pre-orders",
    "Bundle high-end products with exclusive limited-time collections",
    "Highlight brand values and storytelling in marketing materials"
    ],
    "product_strategy": "Showcase luxury items, premium collections, and exclusive limited-edition products."
    },

    "size": "1019 customers",
    "img": "https://cdn-icons-png.flaticon.com/512/18332/18332021.png",
    "size": "272 customers (7.0% of total customers)",
    "purchase_avg": "$58.74",
    "conversion_rate": "3.5%",
    "revenue_increase": "$60K",
    "purchase_frequency": "Weekly (272)",
    "previous_purchases": "14.0",
    "avg_rating": "3.6/5",
    "discount_usage": "39.3%",
    "promo_code_usage": "39.3%",
    "top_categories": ["Clothing", "Accessories"],
    "marketing_recommendations": [
    "Test various engagement strategies to refine segmentation",
    "Develop targeted outreach campaigns to drive conversions",
    "Analyze buying behavior for deeper insights into shopping patterns",
    "Implement automated re-engagement emails and promotions",
    "Create A/B tests for personalized product recommendations"
    ],
    "product_strategy": "Refine product recommendations based on customer behavior and purchase history."
}
}

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

# 📌 Show details when a persona is selected
if selected_persona:
persona = persona_details[selected_persona]
st.subheader(selected_persona)
st.image(persona["img"], width=100)
st.write(f"**Size:** {persona['size']}")
st.write(f"**Average Purchase:** {persona['purchase_avg']}")
st.write(f"**Conversion Rate:** {persona['conversion_rate']}")
st.write(f"**Expected Revenue Increase:** {persona['revenue_increase']}")
st.write(f"**Purchase Frequency:** {persona['purchase_frequency']}")
st.write(f"**Previous Purchases:** {persona['previous_purchases']}")
st.write(f"**Average Rating:** {persona['avg_rating']}")
st.write(f"**Discount Usage:** {persona['discount_usage']}")
st.write(f"**Promo Code Usage:** {persona['promo_code_usage']}")
st.write(f"**Top Categories:** {', '.join(persona['top_categories'])}")
st.write("### 📌 Marketing Recommendations")
for rec in persona["marketing_recommendations"]:
st.write(f"- {rec}")
st.write(f"### 🎯 Product Strategy: {persona['product_strategy']}")

# 📊 Generate Graphs for the Selected Persona
metrics = {
"Recency": [30, 20, 25, 28, 35],
"Frequency": [10, 15, 8, 5, 7],
"Monetary": [80, 60, 50, 75, 45],
"Engagement_Score": [5, 10, 7, 12, 8]
}
metric_df = pd.DataFrame(metrics, index=list(persona_details.keys())).T.reset_index()
metric_df.columns = ["Metric", *persona_details.keys()]

st.write("### 📊 Customer Segment Metrics")
fig = px.bar(metric_df, x="Metric", y=selected_persona, text=selected_persona, title=f"Key Metrics for {selected_persona}", labels={"index": "Metrics", "value": "Mean Value"})
fig.update_traces(texttemplate='%{text}', textposition='outside')
st.plotly_chart(fig)

st.write("### 📌 Business Impact")
st.write("- Personalized marketing campaigns based on segments can enhance conversions.")
st.write("- GMM segmentation provides data-driven customer targeting.")
st.write("- By implementing targeted strategies, Acme Inc. aims to increase revenue by $500K annually.")
