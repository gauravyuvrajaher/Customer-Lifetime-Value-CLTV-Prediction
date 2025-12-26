# Customer-Lifetime-Value-CLTV-Prediction
This project focuses on estimating Customer Lifetime Value (CLTV) using historical transaction data from a Brazilian e-commerce platform. The goal is to help businesses understand which customers are likely to be most valuable in the future, so they can plan marketing, retention, and revenue strategies more effectively.

Rather than relying only on past revenue, this project uses probabilistic customer lifetime models that are widely used in industry and research.

🔍 Business Problem

For medium- to long-term planning, businesses need answers to questions like:

Which customers are likely to purchase again?

How often will they purchase?

How valuable will they be over the next few months?

This project addresses those questions by forecasting future customer behaviour and value based on past purchasing patterns.

🧠 Approach & Methodology

The solution is built using two complementary probabilistic models:

BG-NBD (Beta-Geometric / Negative Binomial Distribution)
Used to predict how frequently a customer is expected to make future purchases.

Gamma-Gamma Model
Used to estimate the expected monetary value of each transaction, assuming spending behaviour is independent of purchase frequency.

By combining both models, CLTV is calculated across multiple time horizons (3, 6, and 12 months), providing flexible insights for different business needs.

Key steps include:

Data cleaning and aggregation at order level

RFM-style feature engineering (Recency, Frequency, Monetary, customer age)

Probabilistic CLTV modelling

Customer segmentation into value tiers

📈 Dashboard & Insights

An interactive Streamlit dashboard is used to present the results in a business-friendly way. The app includes:

Executive-level KPIs

CLTV forecasts for different time horizons

Customer segmentation (Low, Mid, High, VIP)

Expected future purchase behaviour

Model diagnostics and customer exploration

The dashboard is designed so that non-technical users can easily interpret and use the insights.

🧩 Dataset

The analysis is based on a Brazilian e-commerce dataset containing:

~100,000 orders (2016–2018)

Order timestamps

Customer identifiers

Payment values

The data reflects real-world, non-contractual customer purchasing behaviour.

▶️ How to Run the App
pip install -r requirements.txt
streamlit run CLTV_app.py


The app can be run locally or deployed on Streamlit Cloud.

💼 What This Project Demonstrates

This project showcases:

Strong understanding of customer analytics and CLTV

Use of industry-standard probabilistic models

Ability to translate analytics into business value

End-to-end ownership from data preparation to deployment

Clear communication of insights through an interactive dashboard

📌 Author

Gaurav Aher
Business Analytics & Data Science
