import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="CommerceIQ | Sales Forecast", page_icon="📈", layout="wide")

# ---- Data ingestion and normalization ----
ALIASES = {
    "date": ["date", "order_date", "invoicedate", "transaction_date", "timestamp", "day"],
    "quantity": ["quantity", "qty", "units", "units_sold"],
    "price": ["unitprice", "unit_price", "price", "selling_price"],
    "revenue": ["revenue", "sales", "sales_amount", "total_sales", "amount", "turnover"],
    "product": ["product", "product_name", "item", "stockcode", "sku"],
    "category": ["category", "product_category", "department", "class"],
}

def demo_data():
    rng = np.random.default_rng(19)
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=540)
    products = [("Everyday Tee", "Apparel", 24), ("Trail Bottle", "Outdoors", 19),
                ("Desk Lamp", "Home", 42), ("Wireless Mouse", "Electronics", 35),
                ("Canvas Tote", "Accessories", 16), ("Travel Mug", "Outdoors", 28)]
    rows = []
    for j, (name, category, price) in enumerate(products):
        trend = np.linspace(0.8, 1.32 + 0.08 * np.sin(j), len(dates))
        weekly = np.where(dates.dayofweek < 5, 1.08, 0.82)
        annual = 1 + 0.18 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + j / 2)
        promo = np.ones(len(dates))
        promo[rng.choice(len(dates), 18, replace=False)] *= rng.uniform(1.2, 1.65, 18)
        units = rng.poisson((8 + j * 1.6) * trend * weekly * annual * promo)
        for d, u in zip(dates, units):
            rows.append((d, name, category, int(u), price, int(u) * price))
    return pd.DataFrame(rows, columns=["date", "product", "category", "quantity", "unit_price", "revenue"])

def normalize(raw):
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {}
    for target, names in ALIASES.items():
        for col in df.columns:
            if col in names:
                rename[col] = target
                break
    df = df.rename(columns=rename)
    if "date" not in df:
        raise ValueError("A date column is required (for example Date, InvoiceDate, or Order Date).")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"])
    for col in ["quantity", "price", "revenue"]:
        if col in df:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    if "revenue" not in df:
        if {"quantity", "price"}.issubset(df.columns):
            df["revenue"] = df["quantity"] * df["price"]
        else:
            raise ValueError("Provide a revenue/sales column, or both quantity and unit price columns.")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["revenue"])
    df["product"] = df["product"].fillna("Unspecified") if "product" in df else "All products"
    df["category"] = df["category"].fillna("Uncategorized") if "category" in df else "All categories"
    excluded = int(((df["revenue"] < 0) | (df["quantity"] < 0 if "quantity" in df else False)).sum())
    df = df[(df["revenue"] >= 0) & ((df["quantity"] >= 0) if "quantity" in df else True)]
    if df.empty:
        raise ValueError("No valid non-negative sales rows remain after cleaning.")
    return df, excluded

def features(series):
    x = pd.DataFrame(index=series.index)
    x["dow"] = series.index.dayofweek
    x["month"] = series.index.month
    x["dayofyear"] = series.index.dayofyear
    x["trend"] = np.arange(len(series))
    for lag in [1, 7, 14, 28]:
        x[f"lag_{lag}"] = series.shift(lag)
    x["rolling_7"] = series.shift(1).rolling(7).mean()
    x["rolling_28"] = series.shift(1).rolling(28).mean()
    return x

def seasonal_predict(train, dates):
    by_day = train.groupby(train.index.dayofweek).mean()
    fallback = float(train.mean())
    return np.array([by_day.get(d.dayofweek, fallback) for d in dates], dtype=float)

def evaluate(series):
    # Keep at least 28 days in training and use a holdout of up to 56 days.
    horizon = min(56, max(14, int(len(series) * 0.2)))
    split = len(series) - horizon
    if split < 35:
        raise ValueError("At least 50 daily observations are recommended for time-based evaluation.")
    train, test = series.iloc[:split], series.iloc[split:]
    baseline = seasonal_predict(train, test.index)
    x = features(series)
    valid = x.notna().all(axis=1)
    fit_idx = valid & (np.arange(len(series)) < split)
    model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.06, max_leaf_nodes=15,
                                          l2_regularization=2.0, random_state=42)
    model.fit(x.loc[fit_idx], series.loc[fit_idx])
    ml_pred = np.maximum(0, model.predict(x.loc[test.index]))
    def metrics(y, p):
        smape = np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-8)) * 100
        return {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "sMAPE": smape}
    scores = {"Seasonal baseline": metrics(test.values, baseline), "Gradient boosting": metrics(test.values, ml_pred)}
    best_name = min(scores, key=lambda name: scores[name]["MAE"])
    residual = test.values - (baseline if best_name == "Seasonal baseline" else ml_pred)
    return scores, best_name, model, residual

def forecast(series, horizon, best_name, fitted_model):
    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    if best_name == "Seasonal baseline":
        pred = seasonal_predict(series, future_dates)
    else:
        history = series.copy()
        vals = []
        for dt in future_dates:
            row = {"dow": dt.dayofweek, "month": dt.month, "dayofyear": dt.dayofyear, "trend": len(history)}
            for lag in [1, 7, 14, 28]:
                row[f"lag_{lag}"] = history.iloc[-lag] if len(history) >= lag else float(history.mean())
            row["rolling_7"] = history.iloc[-7:].mean()
            row["rolling_28"] = history.iloc[-28:].mean()
            value = max(0.0, float(fitted_model.predict(pd.DataFrame([row]))[0]))
            vals.append(value)
            history.loc[dt] = value
        pred = np.array(vals)
    return pd.DataFrame({"date": future_dates, "forecast": np.maximum(pred, 0)})

# ---- Header and sidebar ----
st.markdown("<div style='font-size:13px;letter-spacing:2px;color:#667085;font-weight:700'>COMMERCEIQ  /  DECISION INTELLIGENCE</div>", unsafe_allow_html=True)
st.title("E-commerce sales forecast")
st.caption("Explore historical performance, compare forecasting models, and plan the next few weeks.")
with st.sidebar:
    st.header("Data & controls")
    upload = st.file_uploader("Upload transaction CSV", type=["csv"])
    st.caption("Required: date + revenue, or date + quantity + unit price.")
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    st.divider()
    st.caption("Demo mode uses clearly labeled synthetic sales data.")
try:
    raw = pd.read_csv(upload) if upload else demo_data()
    df, excluded = normalize(raw)
except Exception as e:
    st.error(f"Could not load this dataset: {e}")
    st.stop()

with st.sidebar:
    cats = sorted(df.category.astype(str).unique())
    selected_cats = st.multiselect("Categories", cats, default=cats)
    products = sorted(df.loc[df.category.astype(str).isin(selected_cats), "product"].astype(str).unique())
    selected_products = st.multiselect("Products", products, default=products[:min(8, len(products))])
filtered = df[df.category.astype(str).isin(selected_cats) & df.product.astype(str).isin(selected_products)]
if filtered.empty:
    st.warning("Choose at least one category and product in the sidebar.")
    st.stop()

start, end = filtered.date.min(), filtered.date.max()
st.caption(f"Data window: **{start:%d %b %Y} – {end:%d %b %Y}** · {len(filtered):,} transactions · {excluded:,} returns/negative rows excluded")

# ---- KPI row ----
span = max((end - start).days + 1, 1)
revenue = filtered.revenue.sum()
units = filtered.quantity.sum() if "quantity" in filtered else np.nan
orders = filtered["invoice"].nunique() if "invoice" in filtered else len(filtered)
prev_start = end - pd.Timedelta(days=min(27, span - 1))
prev = filtered[filtered.date >= prev_start].revenue.sum()
prior = filtered[(filtered.date >= prev_start - pd.Timedelta(days=28)) & (filtered.date < prev_start)].revenue.sum()
growth = ((prev - prior) / prior * 100) if prior else np.nan
k1,k2,k3,k4 = st.columns(4)
k1.metric("Net revenue", f"${revenue:,.0f}", f"{growth:+.1f}% vs prior 28d" if np.isfinite(growth) else None)
k2.metric("Average daily revenue", f"${revenue/span:,.0f}")
k3.metric("Units sold", f"{units:,.0f}" if np.isfinite(units) else "Not provided")
k4.metric("Transactions", f"{orders:,}")

# ---- Trends ----
tab_overview, tab_forecast, tab_data = st.tabs(["Overview", "Forecast & model", "Data quality"])
with tab_overview:
    daily = filtered.groupby("date").revenue.sum().asfreq("D", fill_value=0)
    fig = px.area(x=daily.index, y=daily.values, labels={"x":"Date", "y":"Revenue"}, title="Daily revenue")
    fig.update_traces(line_color="#315efb", fillcolor="rgba(49,94,251,.12)")
    fig.update_layout(margin=dict(l=10,r=10,t=55,b=10), height=340, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        catsales = filtered.groupby("category", as_index=False).revenue.sum().sort_values("revenue", ascending=False)
        figc = px.bar(catsales, x="revenue", y="category", orientation="h", title="Revenue by category", color_discrete_sequence=["#315efb"])
        figc.update_layout(height=330, margin=dict(l=10,r=10,t=55,b=10), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(figc, use_container_width=True)
    with right:
        top = filtered.groupby("product", as_index=False).revenue.sum().nlargest(8, "revenue")
        figp = px.bar(top, x="revenue", y="product", orientation="h", title="Top products by revenue", color_discrete_sequence=["#12b886"])
        figp.update_layout(height=330, margin=dict(l=10,r=10,t=55,b=10), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(figp, use_container_width=True)

with tab_forecast:
    daily = filtered.groupby("date").revenue.sum().asfreq("D", fill_value=0)
    if len(daily) < 50:
        st.warning(f"This selection has {len(daily)} days. At least 50 daily observations are recommended to compare models.")
    else:
        with st.spinner("Evaluating models on a chronological holdout…"):
            scores, best, model, residual = evaluate(daily)
            future = forecast(daily, horizon, best, model)
        a,b,c = st.columns(3)
        for col, name in zip([a,b,c], ["Seasonal baseline", "Gradient boosting", "Selected model"]):
            if name == "Selected model":
                col.metric(name, best)
            else:
                col.metric(f"{name} MAE", f"${scores[name]['MAE']:,.0f}")
        band = float(np.quantile(np.abs(residual), .9))
        future["lower"] = np.maximum(0, future.forecast - band)
        future["upper"] = future.forecast + band
        history_plot = daily.tail(120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_plot.index, y=history_plot.values, name="Actual", line=dict(color="#344054", width=2)))
        fig.add_trace(go.Scatter(x=future.date, y=future.upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future.date, y=future.lower, fill="tonexty", fillcolor="rgba(49,94,251,.14)", line=dict(width=0), name="Approx. 90% band"))
        fig.add_trace(go.Scatter(x=future.date, y=future.forecast, name="Forecast", line=dict(color="#315efb", width=3, dash="dash")))
        fig.update_layout(title=f"Next {horizon} days · selected model: {best}", height=390, margin=dict(l=10,r=10,t=55,b=10), plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)
        score_df = pd.DataFrame(scores).T.rename_axis("Model").reset_index()
        st.subheader("Chronological holdout comparison")
        st.dataframe(score_df.style.format({"MAE":"${:,.2f}", "RMSE":"${:,.2f}", "sMAPE":"{:.1f}%"}), hide_index=True, use_container_width=True)
        st.caption("The model was evaluated on the most recent 14–56 days. Forecast bands use the 90th percentile of absolute holdout residuals and are approximate.")
        export = future.copy()
        export["model"] = best
        st.download_button("Download forecast CSV", export.to_csv(index=False).encode("utf-8"), "sales_forecast.csv", "text/csv")

with tab_data:
    st.subheader("Data quality summary")
    q1,q2,q3 = st.columns(3)
    q1.metric("Valid rows", f"{len(filtered):,}")
    q2.metric("Date coverage", f"{filtered.date.nunique():,} days")
    q3.metric("Excluded return rows", f"{excluded:,}")
    st.caption("The table shows cleaned rows after current filters. Revenue is calculated from quantity × unit price when no revenue field was supplied.")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download cleaned data", filtered.to_csv(index=False).encode("utf-8"), "cleaned_sales_data.csv", "text/csv")
