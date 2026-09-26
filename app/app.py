import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# --- Build robust absolute paths based on this script's own location ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUTS_DIR = os.path.join(BASE_DIR, '..', 'outputs')

# --- Page config ---
st.set_page_config(page_title="Sales Demand Forecasting Dashboard", layout="wide")

# --- Load data and model ---

# --- Load data and model ---
@st.cache_data
def load_data():
    df_full = pd.read_csv(os.path.join(DATA_DIR, 'train_cleaned.csv'), parse_dates=['date'])
    df_features = pd.read_csv(os.path.join(DATA_DIR, 'train_features.csv'), parse_dates=['date'])
    inventory_plan = pd.read_csv(os.path.join(OUTPUTS_DIR, 'inventory_plan.csv'))
    model_comparison = pd.read_csv(os.path.join(OUTPUTS_DIR, 'model_comparison.csv'))
    return df_full, df_features, inventory_plan, model_comparison

@st.cache_resource
def load_model():
    with open(os.path.join(OUTPUTS_DIR, 'xgb_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model

df_full, df, inventory_plan, model_comparison = load_data()
xgb_model = load_model()

# --- Title ---
st.title("📊 Sales Demand Forecasting & Inventory Optimization")
st.markdown("End-to-end demand forecasting using XGBoost, with inventory recommendations.")

# --- Sidebar filters (global, apply across tabs) ---
st.sidebar.header("Filters")
selected_store = st.sidebar.selectbox("Select Store", options=["All"] + sorted(df_full['store'].unique().tolist()))
selected_item = st.sidebar.selectbox("Select Item", options=["All"] + sorted(df_full['item'].unique().tolist()))

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📅 Trends", "🎯 Model Performance", "📦 Inventory"])

# ============ TAB 1: OVERVIEW ============
with tab1:
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Historical Sales", f"{df_full['sales'].sum():,.0f} units")
    with col2:
        st.metric("Avg Daily Sales", f"{df_full['sales'].mean():.1f} units")
    with col3:
        best_model_mae = model_comparison['MAE'].min()
        st.metric("Best Model MAE", f"{best_model_mae:.2f}")
    with col4:
        total_reorder_needed = inventory_plan['reorder_point'].sum()
        st.metric("Total Reorder Point (all products)", f"{total_reorder_needed:,.0f} units")

# ============ TAB 2: TRENDS ============
with tab2:
    st.subheader("Historical Sales Trends")

    filtered_df = df_full.copy()
    if selected_store != "All":
        filtered_df = filtered_df[filtered_df['store'] == selected_store]
    if selected_item != "All":
        filtered_df = filtered_df[filtered_df['item'] == selected_item]

    monthly_trend = filtered_df.set_index('date').resample('ME')['sales'].sum()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly_trend.index, monthly_trend.values, color='#2E86AB', linewidth=1.5)
    ax.set_title(f"Monthly Sales Trend (Store: {selected_store}, Item: {selected_item})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")
    st.pyplot(fig)

# ============ TAB 3: MODEL PERFORMANCE ============
with tab3:
    st.subheader("Model Performance: Actual vs. Predicted Demand")

    df_sorted = df.sort_values(by=['store', 'item', 'date']).reset_index(drop=True)
    split_date = df_sorted['date'].max() - pd.Timedelta(days=90)
    test_data = df_sorted[df_sorted['date'] > split_date].copy()

    feature_cols = ['store', 'item', 'year', 'month', 'day', 'day_of_week',
                     'day_of_year', 'is_weekend', 'sales_lag_1', 'sales_lag_7',
                     'sales_lag_365', 'rolling_mean_7', 'rolling_mean_30']

    test_data['predicted_sales'] = xgb_model.predict(test_data[feature_cols])
    daily_actual_vs_pred = test_data.groupby('date')[['sales', 'predicted_sales']].sum()

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(daily_actual_vs_pred.index, daily_actual_vs_pred['sales'], label='Actual', color='#2E86AB')
    ax2.plot(daily_actual_vs_pred.index, daily_actual_vs_pred['predicted_sales'], label='Predicted', color='#F18F01', linestyle='--')
    ax2.set_title("Actual vs Predicted Demand (Test Period)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Units Sold")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Model Comparison")
    st.dataframe(model_comparison, use_container_width=True)

# ============ TAB 4: INVENTORY ============
with tab4:
    st.subheader("Inventory & Reorder Recommendations")

    inv_filtered = inventory_plan.copy()
    if selected_store != "All":
        inv_filtered = inv_filtered[inv_filtered['store'] == selected_store]
    if selected_item != "All":
        inv_filtered = inv_filtered[inv_filtered['item'] == selected_item]

    st.dataframe(inv_filtered.sort_values(by='reorder_point', ascending=False), use_container_width=True)

    st.markdown("**Top 10 Products by Reorder Priority**")
    top_reorder = inventory_plan.sort_values(by='reorder_point', ascending=False).head(10)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    labels = top_reorder['store'].astype(str) + "-" + top_reorder['item'].astype(str)
    ax3.barh(labels, top_reorder['reorder_point'], color='#A23B72')
    ax3.set_xlabel("Reorder Point (units)")
    ax3.set_ylabel("Store-Item")
    ax3.set_title("Top 10 Products Needing Highest Reorder Points")
    ax3.invert_yaxis()
    st.pyplot(fig3)