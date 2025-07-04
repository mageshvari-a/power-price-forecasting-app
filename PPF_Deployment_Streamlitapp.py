# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load models and scaler
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# App Title
st.title("Power Price Forecasting Dashboard")
st.markdown("Upload your IEX Real-Time Excel data (with MCP) for price prediction using our hybrid model.")

# File Uploader
file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file:
    df = pd.read_excel(file)
    st.write("### Uploaded Data Preview", df.head())
    st.write(f"Initial rows: {df.shape[0]}")

    # Handle Date column
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if not date_cols:
        st.error("Could not find a 'Date' column in uploaded file.")
        st.stop()

    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    df = df.dropna(subset=[date_cols[0]])
    df.rename(columns={date_cols[0]: 'Date'}, inplace=True)

    # Rename columns
    df.rename(columns={
        'Purchase Bid (MWh)': 'Purchase_Bid',
        'Sell Bid (MWh)': 'Sell_Bid',
        'MCV (MWh)': 'MCV',
        'Final Scheduled Volume (MWh)': 'Scheduled_Volume',
        'MCP (Rs/MWh)*': 'MCP',
        'MCP (Rs/MWh) *': 'MCP'
    }, inplace=True)

    # Clean MCP column
    df['MCP'] = df['MCP'].astype(str).str.replace(',', '', regex=False)
    df['MCP'] = pd.to_numeric(df['MCP'], errors='coerce')
    st.write(f"MCP nulls: {df['MCP'].isnull().sum()}")

    if 'Weighted_MCP' in df.columns:
        df.drop(columns=['Weighted_MCP'], inplace=True)

    # Feature Engineering
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df['Day'] = df['Date'].dt.day
    df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)

    df['MCP_lag1'] = df['MCP'].shift(1)
    df['MCP_lag3'] = df['MCP'].shift(3)
    df['MCP_lag7'] = df['MCP'].shift(7)
    df['MCP_volatility7'] = df['MCP'].rolling(window=7).std()
    df['MCP_roll7'] = df['MCP'].rolling(window=7).mean().shift(1)
    df['Purchase_Bid_lag1'] = df['Purchase_Bid'].shift(1)

    # Define features used during training
    features = [
        'Purchase_Bid', 'Sell_Bid', 'MCV', 'Scheduled_Volume',
        'Month', 'Weekday', 'Day',
        'Purchase_Bid_lag1', 'MCP_lag1', 'MCP_lag3', 'MCP_lag7',
        'MCP_volatility7', 'MCP_roll7', 'Is_Weekend'
    ]

    # Convert to numeric
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill nulls for short data (e.g., <14 rows)
    if len(df) < 14:
        st.warning("Less than 14 rows — filling lag/rolling values with 0.")
        df[features] = df[features].fillna(0)
    else:
        df = df.dropna(subset=features)

    if df.empty:
        st.error("Data is empty after preprocessing. Please upload a valid file.")
        st.stop()

    # Scaling
    X = df[features]
    X_scaled = scaler.transform(X)

    # Predict using Hybrid Model
    preds_rf = rf_model.predict(X_scaled)
    preds_xgb = xgb_model.predict(X_scaled)
    df['Predicted_MCP'] = (preds_rf + preds_xgb) / 2

    st.success("Prediction complete!")

    # Show prediction results
    st.write("###Prediction Results", df[['Date', 'MCP', 'Predicted_MCP']])
    st.line_chart(df.set_index('Date')[['MCP', 'Predicted_MCP']])

    # Error plot
    df['Error'] = df['MCP'] - df['Predicted_MCP']
    st.subheader("Prediction Error Histogram")
    plt.figure(figsize=(6, 4))
    plt.hist(df['Error'], bins=20, color='orange', edgecolor='black')
    st.pyplot(plt.gcf())

    # Scatter plot
    st.subheader("Actual vs Predicted MCP")
    plt.figure(figsize=(6, 4))
    plt.scatter(df['MCP'], df['Predicted_MCP'], alpha=0.6, color='teal')
    plt.xlabel("Actual MCP")
    plt.ylabel("Predicted MCP")
    plt.title("Actual vs Predicted MCP")
    st.pyplot(plt.gcf())

    # Download button
    st.download_button(
        label="Download Predictions as CSV",
        data=df[['Date', 'MCP', 'Predicted_MCP']].to_csv(index=False),
        file_name="power_price_predictions.csv",
        mime="text/csv"
    )