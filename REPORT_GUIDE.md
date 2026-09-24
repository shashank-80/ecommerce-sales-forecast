# Project report starter

## Title
E-Commerce Sales Forecasting and Business Analytics Dashboard

## Abstract
This project turns transaction-level retail data into daily revenue insights and short-term forecasts. The dashboard supports CSV ingestion, data cleaning, exploratory analysis, chronological model validation, and downloadable forecasts. Replace this section with measured results from your chosen real dataset.

## Objectives
- Summarize revenue, daily sales, product contribution, and category contribution.
- Prepare transaction data consistently and document excluded/invalid records.
- Compare a day-of-week seasonal baseline with a gradient-boosting regressor.
- Evaluate models using chronological holdout MAE, RMSE, and sMAPE.
- Present an interactive forecast with uncertainty guidance for business planning.

## Methodology
1. Map source columns to date, quantity, unit price, revenue, product, and category.
2. Parse dates, coerce numeric fields, calculate revenue if needed, and exclude returns from positive-sales forecasting.
3. Aggregate revenue by day; represent dates without sales as zero.
4. Reserve the latest 14–56 days for holdout evaluation, preserving chronological order.
5. Compare day-of-week mean (seasonal naive) to HistGradientBoosting with calendar, lag, and rolling features.
6. Select the lower-MAE model, refit on all observations, and forecast the configured horizon.

## Evaluation
Record the dataset, date window, row counts, train/holdout sizes, model scores, and forecast horizon here. Interpret errors in the scale of the business: an MAE of 500 means the typical absolute daily miss is about 500 currency units. Avoid claiming the demo's synthetic-data results generalize to a real retailer.

## Discussion and limitations
The model has no explicit promotion, stock availability, price elasticity, or holiday inputs. Missing days are treated as zero sales, which can be wrong when data ingestion is incomplete. Forecast intervals are approximate residual bands, not calibrated statistical confidence intervals. The dashboard should inform human planning rather than trigger automatic purchasing.

## Future enhancements
- Add holiday, promotion, price, and inventory features.
- Add rolling-origin backtesting and per-category forecasts.
- Add model monitoring and drift alerts.
- Connect to an authenticated data warehouse and role-based access controls.
