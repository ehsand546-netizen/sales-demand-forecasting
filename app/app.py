import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# --- Build robust absolute paths based on this script's own location ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUTS_DIR = os.path.join(BASE_DIR, '..', 'outputs')

# --- Page config ---
st.set_page_config(page_title="Sales Demand Forecasting Dashboard", layout="wide")

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

try:
    df_full, df, inventory_plan, model_comparison = load_data()
    xgb_model = load_model()
except Exception as e:
    st.error("Something went wrong loading the data or model. Please check back later.")
    st.stop()

feature_cols_global = ['store', 'item', 'year', 'month', 'day', 'day_of_week',
                        'day_of_year', 'is_weekend', 'sales_lag_1', 'sales_lag_7',
                        'sales_lag_365', 'rolling_mean_7', 'rolling_mean_30']

st.title("Sales Demand Forecasting & Inventory Optimization")
st.markdown("End-to-end demand forecasting using XGBoost, with inventory recommendations.")

with st.expander("How this dashboard works"):
    st.markdown("""
    **Model**: XGBoost trained on 5 years of daily sales data (2013-2017), 10 stores x 50 items.

    **Features used**: calendar features (month, day-of-week), lag features (sales 1/7/365 days ago),
    and rolling averages (7-day, 30-day) - all computed without data leakage using grouped, shifted operations.

    **Validation**: chronological train/test split (last 90 days held out as unseen test data).

    **Inventory logic**: Safety Stock = Z-score(service level) x demand std x sqrt(lead time).
    Reorder Point = (avg daily demand x lead time) + safety stock. Lead time and service level
    are assumptions (adjustable in the Inventory tab) since the dataset lacks real supplier data.
    """)

st.sidebar.header("Filters")
selected_store = st.sidebar.selectbox("Select Store", options=["All"] + sorted(df_full['store'].unique().tolist()))
selected_item = st.sidebar.selectbox("Select Item", options=["All"] + sorted(df_full['item'].unique().tolist()))

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Model Performance", "Inventory"])

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
        st.metric("Total Reorder Point", f"{total_reorder_needed:,.0f} units")

with tab2:
    st.subheader("Historical Sales Trends")
    filtered_df = df_full.copy()
    if selected_store != "All":
        filtered_df = filtered_df[filtered_df['store'] == selected_store]
    if selected_item != "All":
        filtered_df = filtered_df[filtered_df['item'] == selected_item]

    if filtered_df.empty:
        st.warning("No data available for this selection.")
    else:
        monthly_trend = filtered_df.set_index('date').resample('ME')['sales'].sum()
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(monthly_trend.index, monthly_trend.values, color='#2E86AB', linewidth=1.5)
        ax.set_title(f"Monthly Sales Trend (Store: {selected_store}, Item: {selected_item})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Units Sold")
        st.pyplot(fig)

with tab3:
    st.subheader("Model Performance: Actual vs. Predicted Demand")
    df_sorted = df.sort_values(by=['store', 'item', 'date']).reset_index(drop=True)
    split_date = df_sorted['date'].max() - pd.Timedelta(days=90)
    test_data = df_sorted[df_sorted['date'] > split_date].copy()
    test_data['predicted_sales'] = xgb_model.predict(test_data[feature_cols_global])
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
    st.dataframe(model_comparison, width='stretch')

    st.markdown("---")
    st.subheader("Forecast Next 30 Days")

    forecast_store = st.selectbox("Store for forecast", options=sorted(df['store'].unique()), key="fc_store")
    forecast_item = st.selectbox("Item for forecast", options=sorted(df['item'].unique()), key="fc_item")

    if st.button("Generate 30-Day Forecast"):
        history = df[(df['store'] == forecast_store) & (df['item'] == forecast_item)].sort_values('date').copy()

        if history.empty:
            st.warning("No historical data available for this store-item combination.")
        else:
            last_date = history['date'].max()
            recent_sales = history.set_index('date')['sales'].to_dict()
            future_preds = []
            for i in range(1, 31):
                future_date = last_date + pd.Timedelta(days=i)
                lag_1 = recent_sales.get(future_date - pd.Timedelta(days=1), history['sales'].iloc[-1])
                lag_7 = recent_sales.get(future_date - pd.Timedelta(days=7), history['sales'].mean())
                lag_365 = recent_sales.get(future_date - pd.Timedelta(days=365), history['sales'].mean())
                last_7_vals = [recent_sales.get(future_date - pd.Timedelta(days=d), history['sales'].mean()) for d in range(1, 8)]
                last_30_vals = [recent_sales.get(future_date - pd.Timedelta(days=d), history['sales'].mean()) for d in range(1, 31)]

                row = pd.DataFrame([{
                    'store': forecast_store, 'item': forecast_item,
                    'year': future_date.year, 'month': future_date.month, 'day': future_date.day,
                    'day_of_week': future_date.dayofweek, 'day_of_year': future_date.dayofyear,
                    'is_weekend': int(future_date.dayofweek >= 5),
                    'sales_lag_1': lag_1, 'sales_lag_7': lag_7, 'sales_lag_365': lag_365,
                    'rolling_mean_7': np.mean(last_7_vals), 'rolling_mean_30': np.mean(last_30_vals)
                }])

                pred = xgb_model.predict(row[feature_cols_global])[0]
                recent_sales[future_date] = pred
                future_preds.append({'date': future_date, 'forecasted_sales': pred})

            future_df = pd.DataFrame(future_preds)

            fig4, ax4 = plt.subplots(figsize=(14, 4))
            ax4.plot(history['date'].tail(60), history['sales'].tail(60), label='Recent Actual', color='#2E86AB')
            ax4.plot(future_df['date'], future_df['forecasted_sales'], label='30-Day Forecast', color='#F18F01', linestyle='--')
            ax4.set_title(f"30-Day Forward Forecast - Store {forecast_store}, Item {forecast_item}")
            ax4.legend()
            st.pyplot(fig4)
            st.dataframe(future_df, width='stretch')

with tab4:
    st.subheader("Inventory & Reorder Recommendations")

    st.markdown("**Adjust Assumptions**")
    col_a, col_b = st.columns(2)
    with col_a:
        user_lead_time = st.slider("Lead Time (days)", min_value=1, max_value=30, value=7)
    with col_b:
        user_service_level = st.slider("Service Level (%)", min_value=80, max_value=99, value=95)

    user_z = norm.ppf(user_service_level / 100)

    inv_live = inventory_plan.copy()
    inv_live['safety_stock'] = (user_z * inv_live['demand_std'] * np.sqrt(user_lead_time)).round().astype(int)
    inv_live['reorder_point'] = (
        (inv_live['forecasted_avg_daily_demand'] * user_lead_time) + inv_live['safety_stock']
    ).round().astype(int)

    st.caption(f"Using Z-score {user_z:.2f} for {user_service_level}% service level")

    inv_filtered = inv_live.copy()
    if selected_store != "All":
        inv_filtered = inv_filtered[inv_filtered['store'] == selected_store]
    if selected_item != "All":
        inv_filtered = inv_filtered[inv_filtered['item'] == selected_item]

    if inv_filtered.empty:
        st.warning("No data available for this selection.")
    else:
        st.dataframe(inv_filtered.sort_values(by='reorder_point', ascending=False), width='stretch')

    st.markdown("**Top 10 Products by Reorder Priority**")
    top_reorder = inv_live.sort_values(by='reorder_point', ascending=False).head(10)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    labels = top_reorder['store'].astype(str) + "-" + top_reorder['item'].astype(str)
    ax3.barh(labels, top_reorder['reorder_point'], color='#A23B72')
    ax3.set_xlabel("Reorder Point (units)")
    ax3.set_ylabel("Store-Item")
    ax3.set_title("Top 10 Products Needing Highest Reorder Points")
    ax3.invert_yaxis()
    st.pyplot(fig3)