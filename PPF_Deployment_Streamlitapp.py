# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load model & scaler
model = joblib.load("hybrid_xgb_rf_models.pkl")  # XGBoost + RF Hybrid in our case
scaler = joblib.load("scaler.pkl")     # You need to save your scaler too

# App title
st.title("Power Price Forecasting Dashboard")
st.markdown("Upload your IEX Real-Time Excel data to view MCP predictions.")

# File uploader
file = st.file_uploader("Upload Excel file", type=["xlsx"])

if file:
    df = pd.read_excel(file)
    st.write("### Uploaded Data Preview", df.head())

    # Try to automatically locate 'Date' column
    possible_date_cols = [col for col in df.columns if 'date' in col.lower()]
    if possible_date_cols:
        df['Date'] = pd.to_datetime(df[possible_date_cols[0]], errors='coerce')
    else:
        st.error("Could not find a 'Date' column in the uploaded Excel file.")
        st.stop()

   # Rename columns properly
    df.rename(columns={
        'Purchase Bid (MWh)': 'Purchase_Bid',
        'Sell Bid (MWh)': 'Sell_Bid',
        'MCV (MWh)': 'MCV',
        'Final Scheduled Volume (MWh)': 'Scheduled_Volume',
        'MCP (Rs/MWh) *': 'MCP'
    }, inplace=True)

    # Confirm renamed columns
    st.write("Cleaned Column Names:", df.columns.tolist())

    df['MCP'] = df['MCP'].astype(str).str.replace(',', '', regex=False)
    df['MCP'] = pd.to_numeric(df['MCP'], errors='coerce')

    # Drop Weighted_MCP if it exists
    if 'Weighted_MCP' in df.columns:
        df.drop(columns=['Weighted_MCP'], inplace=True)

    # Create required features
    df['Date'] = pd.to_datetime(df['Date'])
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

    df.dropna(inplace=True)

    # Define features
    features = [
        'Purchase_Bid', 'Sell_Bid', 'MCV', 'Scheduled_Volume',
        'Month', 'Weekday', 'Day',
        'Purchase_Bid_lag1', 'MCP_lag1', 'MCP_lag3', 'MCP_lag7',
        'MCP_volatility7', 'MCP_roll7', 'Is_Weekend'
    ]

    # Ensure all feature columns are numeric
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with missing feature values after conversion
    df = df.dropna(subset=features)

    # If nothing left after cleaning, show error and stop
    if df.empty:
        st.error("Data after preprocessing is empty. Please check your file format or contents.")
        st.stop()
    
    X = df[features]
    X_scaled = scaler.transform(X)

    # Predict
    df['Predicted_MCP'] = model.predict(X_scaled)

    st.success("Prediction completed successfully!")

    # Show results
    st.write("### Prediction Results", df[['Date', 'MCP', 'Predicted_MCP']])

    # Plots
    st.line_chart(df.set_index('Date')[['MCP', 'Predicted_MCP']])
    
    st.subheader("Prediction Error Distribution")
    df['Error'] = df['MCP'] - df['Predicted_MCP']
    plt.figure(figsize=(6, 4))
    plt.hist(df['Error'], bins=30, color='orange', edgecolor='black')
    plt.title("Error Distribution (Actual - Predicted MCP)")
    st.pyplot(plt.gcf())

    st.subheader("Actual vs Predicted Scatter Plot")
    plt.figure(figsize=(6, 4))
    plt.scatter(df['MCP'], df['Predicted_MCP'], alpha=0.6, color='teal')
    plt.xlabel("Actual MCP")
    plt.ylabel("Predicted MCP")
    plt.title("Actual vs Predicted MCP")
    st.pyplot(plt.gcf())

    st.download_button(
    label="Download Predictions as CSV",
    data=df[['Date', 'MCP', 'Predicted_MCP']].to_csv(index=False),
    file_name="power_price_predictions.csv",
    mime="text/csv"
    )

