import streamlit as st
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

st.set_page_config(
    page_title="Advanced CLTV Analytics",
    page_icon="📊",
    layout="wide"
)

# Sidebar

st.sidebar.title("📊 CLTV Controls")
analysis_date = st.sidebar.date_input(
    "Analysis Date",
    value=dt.date(2018, 9, 5)
)

discount_rate = st.sidebar.slider(
    "Discount Rate",
    0.0, 0.05, 0.01, 0.005
)


# Title

st.title("📈 Customer Lifetime Value (CLTV) Analytics")
st.markdown(
    """
    **Probabilistic CLTV Modeling using BG-NBD & Gamma-Gamma**  
    Dataset: Brazilian E-commerce Transactions (2016–2018)
    """
)


# Load Data
@st.cache_data
def load_data():
    orders = pd.read_csv(
        "https://raw.githubusercontent.com/gauravyuvrajaher/Customer-Lifetime-Value-CLTV-Prediction/main/order.csv",
        sep=",",
        engine="python"
    )

    customers = pd.read_csv(
        "https://raw.githubusercontent.com/gauravyuvrajaher/Customer-Lifetime-Value-CLTV-Prediction/main/customer.csv",
        sep=",",
        engine="python"
    )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    df = orders.merge(customers, on="order_id", how="inner")

    df = df.groupby(
        ["customer_id", "order_id", "order_purchase_timestamp"],
        as_index=False
    )["payment_value"].sum()

    return df

df = load_data()


# CLTV Feature Engineering

today_date = dt.datetime.combine(analysis_date, dt.datetime.min.time())

cltv_df = df.groupby("customer_id").agg({
    "order_purchase_timestamp": [
        lambda x: (x.max() - x.min()).days,
        lambda x: (today_date - x.min()).days
    ],
    "order_id": "nunique",
    "payment_value": "sum"
})

cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df = cltv_df[cltv_df["frequency"] > 1]

# Weekly conversion
cltv_df["recency"] /= 7
cltv_df["T"] /= 7
cltv_df["monetary"] /= cltv_df["frequency"]


# BG-NBD Model

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(
    cltv_df["frequency"],
    cltv_df["recency"],
    cltv_df["T"]
)

cltv_df["exp_purc_1m"] = bgf.predict(
    4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]
)

cltv_df["exp_purc_3m"] = bgf.predict(
    12, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]
)


# Gamma-Gamma Model

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(
    cltv_df["frequency"],
    cltv_df["monetary"]
)

cltv_df["exp_avg_profit"] = ggf.conditional_expected_average_profit(
    cltv_df["frequency"],
    cltv_df["monetary"]
)


# CLTV Calculation (Multiple Horizons)

for m in [3, 6, 12]:
    cltv_df[f"cltv_{m}m"] = ggf.customer_lifetime_value(
        bgf,
        cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"],
        cltv_df["monetary"],
        time=m,
        freq="W",
        discount_rate=discount_rate
    )

# Segmentation

cltv_df["segment"] = pd.qcut(
    cltv_df["cltv_12m"],
    4,
    labels=["Low Value", "Mid Value", "High Value", "VIP"]
)


# Executive KPIs

st.subheader("📌 Executive Summary")
st.info("""
📌 **How to interpret the results**

- CLTV represents the **expected future value** of a customer.
- Higher CLTV indicates customers who are more valuable over time.
- Segments (Low, Mid, High, VIP) are created based on **12-month CLTV**.
- These insights support marketing prioritisation and strategic planning.
""")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", cltv_df.shape[0])
col2.metric("Total 12M CLTV", f"${cltv_df['cltv_12m'].sum():,.0f}")
col3.metric("Avg CLTV (12M)", f"${cltv_df['cltv_12m'].mean():,.0f}")
col4.metric("VIP Revenue Share",
            f"{(cltv_df[cltv_df['segment']=='VIP']['cltv_12m'].sum() / cltv_df['cltv_12m'].sum())*100:.1f}%")

# Segment Analysis

st.subheader("🎯 CLTV Segment Analysis")

segment_summary = cltv_df.groupby("segment").agg({
    "frequency": "mean",
    "exp_purc_3m": "mean",
    "exp_avg_profit": "mean",
    "cltv_12m": ["mean", "sum"]
})

st.dataframe(segment_summary.style.format("{:,.2f}"))


# Visualizations

st.subheader("📊 Visual Insights")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(
        x=cltv_df["segment"].value_counts().index,
        y=cltv_df["segment"].value_counts().values,
        ax=ax
    )
    ax.set_title("Customer Distribution by Segment")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(
        x="segment",
        y="cltv_12m",
        data=cltv_df,
        ax=ax
    )
    ax.set_title("CLTV Distribution by Segment")
    st.pyplot(fig)


# Model Diagnostics

st.subheader("🧠 BG-NBD Model Diagnostics")

fig, ax = plt.subplots()
plot_period_transactions(bgf, ax=ax)
st.pyplot(fig)


# Customer Explorer

st.subheader("🔍 Customer Explorer")

selected_segment = st.selectbox(
    "Select Segment",
    cltv_df["segment"].unique()
)

st.dataframe(
    cltv_df[cltv_df["segment"] == selected_segment]
    .sort_values("cltv_12m", ascending=False)
    .head(20)
)


# Footer

st.markdown(
    """
    ---
    **Built with BG-NBD & Gamma-Gamma | CLTV for Strategic Decision Making**  
    Ideal for Marketing, Retention & Revenue Forecasting
    """
)








