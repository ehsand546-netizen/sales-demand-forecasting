# 📊 Intelligent Sales Demand Forecasting & Inventory Optimization

An end-to-end data science project that forecasts product demand using machine learning and translates those forecasts into actionable inventory recommendations.

## 🎯 Project Overview

This project builds a complete pipeline — from raw sales data to a deployed, interactive dashboard — that:
- Forecasts daily product demand per store using XGBoost
- Compares multiple forecasting approaches (naive baselines, Linear Regression, Random Forest, XGBoost)
- Translates demand forecasts into inventory decisions (safety stock, reorder points)
- Presents everything through an interactive Streamlit dashboard

## 📂 Dataset

**Source:** [Kaggle - Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)

- 913,000 rows spanning 2013–2017 (5 years, daily granularity)
- 10 stores × 50 items
- Columns: `date`, `store`, `item`, `sales`

## 🛠️ Tech Stack

Python · Pandas · NumPy · SQL (SQLite) · Matplotlib · Seaborn · Scikit-learn · XGBoost · Statsmodels · Streamlit

## 🔄 Project Pipeline

1. **Data Understanding** — verified shape, types, ranges, and completeness
2. **Data Cleaning** — datetime conversion, duplicate/outlier checks (outliers found to be genuine seasonal spikes, retained)
3. **SQL Analysis** — store/item performance, yearly and monthly trend queries
4. **Exploratory Data Analysis** — visualized trend, seasonality, and store/item distributions
5. **Time-Series Analysis** — seasonal decomposition (trend/seasonal/residual), ADF stationarity test
6. **Feature Engineering** — calendar features, lag features (1/7/365 days), rolling averages — all leakage-checked via grouped, shifted operations
7. **Baseline Models** — Naive and Seasonal Naive forecasts established as benchmarks
8. **ML Models** — Linear Regression, Random Forest, XGBoost, all evaluated on a chronological (non-random) train/test split
9. **Model Comparison** — XGBoost selected as production model
10. **Inventory Optimization** — safety stock and reorder point calculations using forecasted demand, demand variability, and standard supply-chain formulas
11. **Streamlit Dashboard** — interactive KPIs, trends, model performance, and inventory recommendations

## 📈 Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive (lag_1) | 10.65 | 14.47 | 22.88% |
| Seasonal Naive (lag_365) | 10.79 | 14.36 | 22.57% |
| Linear Regression | 7.16 | 9.35 | 15.88% |
| Random Forest | 6.15 | 8.02 | 13.49% |
| **XGBoost (final model)** | **5.93** | **7.68** | **13.05%** |

XGBoost reduced forecast error by **~44%** compared to the naive baseline.

## 💡 Key Business Insights

- Strong yearly seasonality: sales peak in July (~67 units/day avg) and trough in Jan/Dec (~35-39 units/day avg)
- Consistent ~35% year-over-year demand growth from 2013 to 2017
- Store performance varies significantly: top store (Store 2) sells ~2x the volume of the weakest store (Store 7)
- Item-level demand spans a 4.8x range between best- and worst-selling products

## ⚠️ Assumptions & Limitations

- **Lead time** (7 days) and **service level** (95%) were assumed, as the dataset does not include supplier/logistics data — clearly documented as assumptions, not derived values
- Dataset lacks promotional, holiday, and pricing data, which likely explain some residual forecast error
- Store/item IDs were used as raw numeric features for simplicity; true categorical encoding could further improve results

## 🚀 Future Improvements

- Incorporate external features (holidays, promotions, weather) if available
- Hyperparameter tuning via GridSearch/Optuna for further model improvement
- Per-store or per-item model specialization instead of one global model
- Real supplier lead-time data integration for more accurate reorder points

## 📁 Repository Structure