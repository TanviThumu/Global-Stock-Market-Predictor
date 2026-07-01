import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Global Stock Predictor", page_icon="📈", layout="centered")

st.title("Global Stock Market Predictor")
st.caption("Next-day price direction prediction using XGBoost")

# ---------------------------------------------------------------------------
# EXTENSIVE COMPANIES DICTIONARY (Global & Indian)
# ---------------------------------------------------------------------------
POPULAR_COMPANIES = {
    # --- US / Global Tech & Comm ---
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Alphabet / Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Meta / Facebook (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "Broadcom (AVGO)": "AVGO",
    "AMD (AMD)": "AMD",
    "Netflix (NFLX)": "NFLX",
    "Intel (INTC)": "INTC",
    "Cisco (CSCO)": "CSCO",
    "Oracle (ORCL)": "ORCL",
    "Salesforce (CRM)": "CRM",
    
    # --- US / Global Finance, Health & Consumer ---
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "JPMorgan Chase (JPM)": "JPM",
    "Visa (V)": "V",
    "Mastercard (MA)": "MA",
    "Walmart (WMT)": "WMT",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Procter & Gamble (PG)": "PG",
    "Exxon Mobil (XOM)": "XOM",
    "UnitedHealth (UNH)": "UNH",
    "Home Depot (HD)": "HD",
    "Coca-Cola (KO)": "KO",
    "PepsiCo (PEP)": "PEP",
    "Costco (COST)": "COST",
    "McDonald's (MCD)": "MCD",
    
    # --- International Giants ---
    "TSMC (TSM)": "TSM",
    "ASML (ASML)": "ASML",
    "Novo Nordisk (NVO)": "NVO",
    "Toyota (TM)": "TM",
    "AstraZeneca (AZN)": "AZN",
    "Shell (SHEL)": "SHEL",
    "Sony (SONY)": "SONY",
    
    # --- Indian / NSE Top Cap (NIFTY) ---
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "State Bank of India (SBIN.NS)": "SBIN.NS",
    "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
    "ITC Limited (ITC.NS)": "ITC.NS",
    "Larsen & Toubro (LT.NS)": "LT.NS",
    "Bajaj Finance (BAJFINANCE.NS)": "BAJFINANCE.NS",
    "Hindustan Unilever (HINDUNILVR.NS)": "HINDUNILVR.NS",
    "Axis Bank (AXISBANK.NS)": "AXISBANK.NS",
    "Kotak Mahindra Bank (KOTAKBANK.NS)": "KOTAKBANK.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "Sun Pharma (SUNPHARMA.NS)": "SUNPHARMA.NS",
    "Maruti Suzuki (MARUTI.NS)": "MARUTI.NS",
    "Asian Paints (ASIANPAINT.NS)": "ASIANPAINT.NS",
    "HCL Technologies (HCLTECH.NS)": "HCLTECH.NS",
    "Titan Company (TITAN.NS)": "TITAN.NS",
    "Adani Enterprises (ADANIENT.NS)": "ADANIENT.NS"
}

# ---------------------------------------------------------------------------
# 1. DATA SOURCE SELECTION & STATE MANAGEMENT
# ---------------------------------------------------------------------------
st.sidebar.header("Data Configuration")
data_source = st.sidebar.radio(
    "Select Data Source:",
    ("Import Market Data", "Upload Custom Dataset"),
)

# Clear session state if the user switches data sources to prevent contamination
if "prev_data_source" in st.session_state and st.session_state["prev_data_source"] != data_source:
    if "raw_df" in st.session_state:
        del st.session_state["raw_df"]
    if "label" in st.session_state:
        del st.session_state["label"]

st.session_state["prev_data_source"] = data_source

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and validate the uploaded/fetched dataframe."""
    col_map = {str(c).lower().strip(): c for c in df.columns}
    rename = {}
    for req in REQUIRED_COLS + ["Date"]:
        if req.lower() in col_map:
            rename[col_map[req.lower()]] = req
    df = df.rename(columns=rename)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Expected columns (case-insensitive): Date, Open, High, Low, Close, Volume."
        )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.set_index("Date")
    else:
        try:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except Exception:
            st.warning("No 'Date' column found — using row order as time sequence.")

    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLS)

    if len(df) < 60:
        raise ValueError(
            f"Dataset only has {len(df)} usable rows after cleaning. "
            f"Need at least ~60 rows of daily data for reliable training."
        )

    return df[REQUIRED_COLS]

# Cached file loader to prevent reprocessing file on every rerun
@st.cache_data
def load_uploaded_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

raw_df = None
ticker_label = "Uploaded Dataset"

if data_source == "Import Market Data":
    # Simple dropdown without custom text input
    selected_display = st.sidebar.selectbox("Select Asset:", list(POPULAR_COMPANIES.keys()))
    ticker = POPULAR_COMPANIES[selected_display]
    
    period = st.sidebar.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"], index=1)

    if st.sidebar.button("Import Data", type="primary"):
        with st.spinner(f"Retrieving market data for {ticker}..."):
            try:
                data = yf.download(ticker, period=period, progress=False)
                if data.empty:
                    st.error("No data returned. The data provider might be temporarily unavailable.")
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    raw_df = validate_and_prepare(data.reset_index())
                    ticker_label = ticker
                    st.session_state["raw_df"] = raw_df
                    st.session_state["label"] = ticker_label
            except Exception as e:
                st.error(f"Error retrieving data: {e}")

else:
    st.sidebar.markdown(
        "Upload a CSV or Excel file with columns:\n"
        "`Date, Open, High, Low, Close, Volume`\n\n"
        "(column names are case-insensitive)"
    )
    uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
            raw_df = validate_and_prepare(data)
            ticker_label = uploaded_file.name
            st.session_state["raw_df"] = raw_df
            st.session_state["label"] = ticker_label
            st.sidebar.success(f"Loaded {len(raw_df)} rows")
        except Exception as e:
            st.sidebar.error(f"Could not process file: {e}")

# Persist across reruns
if raw_df is None and "raw_df" in st.session_state:
    raw_df = st.session_state["raw_df"]
    ticker_label = st.session_state.get("label", ticker_label)

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["Volatility_5"] = df["Return"].rolling(5).std()

    df["HL_Spread"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    df["Vol_Change"] = df["Volume"].replace(0, np.nan).pct_change()

    tomorrow_close = df["Close"].shift(-1)
    df["Target"] = np.where(
        tomorrow_close.isna(),
        np.nan,
        (tomorrow_close > df["Close"]).astype(float)
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

FEATURE_COLS = [
    "Return", "SMA_5", "SMA_10", "EMA_5", "EMA_10",
    "Volatility_5", "HL_Spread", "Vol_Change",
]

# ---------------------------------------------------------------------------
# 3. TRAIN + PREDICT
# ---------------------------------------------------------------------------
if raw_df is not None:
    st.subheader(f"Data Overview — {ticker_label}")
    st.dataframe(raw_df.tail(10), use_container_width=True)
    st.line_chart(raw_df["Close"])

    feat_df = engineer_features(raw_df)

    # Isolate the last row (today) for true next-day prediction BEFORE dropping NaNs
    latest_features = feat_df[FEATURE_COLS].iloc[[-1]]
    latest_features = latest_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Drop NaNs (removes initial rolling windows and the unlabelled final row from training)
    train_df = feat_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna()

    if len(train_df) < 30:
        st.error("Not enough data after feature engineering (need 30+ rows). Try a longer historical period.")
    else:
        X = train_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = train_df["Target"]

        split = int(len(X)*0.8)
        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        with st.spinner("Training predictive model..."):
            model = XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
            )
            if np.isinf(X_train.values).any() or X_train.isnull().sum().sum()>0:
                st.error("Training data contains invalid values.")
                st.stop()
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.subheader("Model Performance")
        st.metric("Backtest Accuracy (last 20% of data)", f"{acc*100:.2f}%")

        # Predict next-day direction using isolated clean row
        next_day_pred = model.predict(latest_features)[0]
        next_day_proba = model.predict_proba(latest_features)[0]
        confidence = max(next_day_proba) * 100

        last_close = raw_df["Close"].iloc[-1]
        recent_volatility = feat_df["Volatility_5"].iloc[-1]
        estimated_change = 0 if pd.isna(recent_volatility) else recent_volatility * last_close
        estimated_price = (
            last_close + estimated_change if next_day_pred == 1 else last_close - estimated_change
        )

        st.subheader("Next-Day Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric("Direction", "UP" if next_day_pred == 1 else "DOWN")
        col2.metric("Confidence", f"{confidence:.1f}%")
        col3.metric("Estimated Price Value", f"{estimated_price:,.2f}")

        with st.expander("Feature Importance"):
            importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
            st.bar_chart(importance)
else:
    st.info("Configure your data source in the sidebar to get started: import market data or upload a custom dataset.")