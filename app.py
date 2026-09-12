import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Demand Forecasting Manufaktur",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Demand Forecasting Bahan Baku")
st.caption("Aplikasi analisis dan forecasting demand untuk data manufaktur")

@st.cache_data
def load_data(file_or_path):
    df = pd.read_csv(file_or_path)

    # Hapus kolom index yang tidak diperlukan
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Pastikan kolom utama tersedia
    required = ["Date", "Warehouse", "Product_Code", "Product_Category", "Order_Demand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    # Cleaning
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Order_Demand"] = (
        df["Order_Demand"]
        .astype(str)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Order_Demand"] = pd.to_numeric(df["Order_Demand"], errors="coerce")

    df = df.dropna(subset=["Date", "Order_Demand"]).copy()
    df = df.sort_values("Date")
    return df

def make_forecast(df):
    monthly = df.set_index("Date").resample("ME")["Order_Demand"].sum().sort_index()
    data = pd.DataFrame({"Demand": monthly})
    data["Lag_1"] = data["Demand"].shift(1)
    data["Lag_2"] = data["Demand"].shift(2)
    data["Lag_3"] = data["Demand"].shift(3)
    data["MA_3"] = data["Demand"].shift(1).rolling(3).mean()
    data = data.dropna()

    if len(data) < 8:
        return monthly, data, None, None, None, None

    X = data[["Lag_1", "Lag_2", "Lag_3", "MA_3"]]
    y = data["Demand"]

    split = max(1, int(len(data) * 0.8))
    if split >= len(data):
        split = len(data) - 1

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    result = pd.DataFrame({
        "Actual": y_test.values,
        "Forecast": pred
    }, index=y_test.index)

    # Forecast one period ahead
    last3 = monthly.iloc[-3:]
    next_features = pd.DataFrame({
        "Lag_1": [monthly.iloc[-1]],
        "Lag_2": [monthly.iloc[-2]],
        "Lag_3": [monthly.iloc[-3]],
        "MA_3": [last3.mean()]
    })
    next_forecast = float(model.predict(next_features)[0])

    return monthly, data, result, mae, rmse, next_forecast

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
uploaded = st.sidebar.file_uploader(
    "Upload dataset CSV",
    type=["csv"],
    help="Upload file CSV dengan kolom Date, Warehouse, Product_Code, Product_Category, dan Order_Demand."
)

default_path = "Demand_Forecasting .csv"

try:
    if uploaded is not None:
        df = load_data(uploaded)
        st.sidebar.success("Dataset upload berhasil dibaca.")
    else:
        df = load_data(default_path)
        st.sidebar.info("Menggunakan dataset bawaan aplikasi.")
except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()

# Sidebar filter
st.sidebar.subheader("🔎 Filter")
warehouses = sorted(df["Warehouse"].dropna().astype(str).unique())
products = sorted(df["Product_Code"].dropna().astype(str).unique())
categories = sorted(df["Product_Category"].dropna().astype(str).unique())

selected_warehouse = st.sidebar.multiselect(
    "Warehouse",
    warehouses,
    default=warehouses
)
selected_product = st.sidebar.multiselect(
    "Product Code",
    products,
    default=products
)
selected_category = st.sidebar.multiselect(
    "Product Category",
    categories,
    default=categories
)

filtered = df[
    df["Warehouse"].astype(str).isin(selected_warehouse)
    & df["Product_Code"].astype(str).isin(selected_product)
    & df["Product_Category"].astype(str).isin(selected_category)
].copy()

# KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Total Data", f"{len(filtered):,}")
c2.metric("📦 Total Demand", f"{filtered['Order_Demand'].sum():,.0f}")
c3.metric("🏭 Produk", f"{filtered['Product_Code'].nunique():,}")
c4.metric("🏢 Warehouse", f"{filtered['Warehouse'].nunique():,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data",
    "📈 Analisis Demand",
    "🔮 Forecasting",
    "💾 Download"
])

with tab1:
    st.subheader("Data yang digunakan")
    st.dataframe(filtered, use_container_width=True, height=450)

    st.subheader("Statistik Order Demand")
    st.dataframe(filtered["Order_Demand"].describe().to_frame().T, use_container_width=True)

with tab2:
    st.subheader("Tren Demand Harian")
    daily = filtered.groupby("Date")["Order_Demand"].sum().sort_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Order Demand")
    ax.set_title("Demand Harian")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Demand Bulanan")
    monthly = filtered.set_index("Date").resample("ME")["Order_Demand"].sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Order Demand")
    ax.set_title("Demand Bulanan")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demand per Produk")
        product_demand = (
            filtered.groupby("Product_Code")["Order_Demand"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(product_demand)

    with col2:
        st.subheader("Demand per Warehouse")
        warehouse_demand = (
            filtered.groupby("Warehouse")["Order_Demand"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(warehouse_demand)

with tab3:
    st.subheader("🔮 Demand Forecasting")

    if len(filtered) < 20:
        st.warning("Data hasil filter terlalu sedikit untuk forecasting yang stabil. Pilih lebih banyak data.")
    else:
        monthly, forecast_data, result, mae, rmse, next_forecast = make_forecast(filtered)

        if result is None:
            st.warning("Data bulanan belum cukup untuk membuat model forecasting.")
        else:
            a, b, c = st.columns(3)
            a.metric("MAE", f"{mae:,.2f}")
            b.metric("RMSE", f"{rmse:,.2f}")
            c.metric("Forecast Periode Berikutnya", f"{next_forecast:,.0f}")

            st.info(
                "Model yang digunakan: Linear Regression dengan fitur Lag 1, Lag 2, "
                "Lag 3, dan Moving Average 3 periode. Data time series dibagi "
                "80% training dan 20% testing tanpa shuffle."
            )

            st.subheader("Actual vs Forecast")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(result.index, result["Actual"], marker="o", label="Actual")
            ax.plot(result.index, result["Forecast"], marker="o", label="Forecast")
            ax.set_xlabel("Bulan")
            ax.set_ylabel("Order Demand")
            ax.set_title("Actual vs Forecast")
            ax.legend()
            ax.grid(True, alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Hasil Forecast pada Data Testing")
            st.dataframe(result, use_container_width=True)

            st.subheader("Forecast Periode Berikutnya")
            next_period = monthly.index[-1] + pd.offsets.MonthEnd(1)
            st.write(
                f"Perkiraan demand untuk **{next_period.strftime('%B %Y')}**: "
                f"**{next_forecast:,.0f} unit**"
            )

with tab4:
    st.subheader("💾 Download Hasil")

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Data Hasil Filter (CSV)",
        csv_data,
        "data_hasil_filter.csv",
        "text/csv"
    )

    monthly_download = (
        filtered.set_index("Date")
        .resample("ME")["Order_Demand"]
        .sum()
        .rename("Demand")
        .reset_index()
    )
    st.download_button(
        "⬇️ Download Demand Bulanan (CSV)",
        monthly_download.to_csv(index=False).encode("utf-8"),
        "demand_bulanan.csv",
        "text/csv"
    )

    try:
        _, _, result, mae, rmse, next_forecast = make_forecast(filtered)
        if result is not None:
            forecast_download = result.reset_index()
            forecast_download.columns = ["Date", "Actual", "Forecast"]
            st.download_button(
                "⬇️ Download Hasil Forecasting (CSV)",
                forecast_download.to_csv(index=False).encode("utf-8"),
                "hasil_forecasting.csv",
                "text/csv"
            )
    except Exception:
        pass

st.divider()
st.caption("Demand Forecasting Bahan Baku — Streamlit")
