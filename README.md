# E-Commerce Sales Forecasting & Analytics

A major-project-ready Streamlit application for analyzing transaction data and forecasting daily revenue. It includes data cleaning, business KPIs, time-based validation, seasonal baselines, a machine-learning model, forecast intervals, and exportable results.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

A built-in demo dataset makes the application usable immediately. Use **Upload your data** in the sidebar to analyze your own CSV.

## Input data

Required: a transaction date column. Optional: quantity, unit price or revenue, product, and category. Common names such as `InvoiceDate`, `Quantity`, `UnitPrice`, `Sales`, `Product`, and `Category` are recognized. If revenue is absent, it is calculated as quantity × unit price. If no product/category exists, an all-products segment is used.

The upload is processed in memory and is not sent to an external service. Returns/negative quantities are excluded from the sales forecast; the app reports how many rows were excluded.

## Forecast method

The app aggregates revenue by day, fills missing dates with zero, and uses a chronological holdout. A day-of-week seasonal-naive model is compared with a HistGradientBoosting model using calendar, lag, and rolling-window features. The better holdout MAE model is refit on all available history for the requested horizon. Forecast bands are approximate and derived from holdout residual error; they are not formal statistical prediction intervals.

## Suggested major-project report sections

1. Problem statement and objectives
2. Dataset and data dictionary
3. Data cleaning and exploratory analysis
4. Time-series feature engineering
5. Baseline and ML model comparison
6. Evaluation with MAE, RMSE, and sMAPE
7. Dashboard design and business insights
8. Limitations, ethics, and future work

## Limitations

This is a forecasting decision-support prototype, not an inventory/replenishment system. It assumes historical patterns remain informative. Promotions, stockouts, holidays, price changes, and new products can materially alter demand. The demonstration data is synthetic and should not be presented as real business data.
