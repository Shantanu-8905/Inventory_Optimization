import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ============================
# 1. PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="📊 Demand Forecasting App",
    layout="wide",
    page_icon="📈"
)

# ============================
# 2. CUSTOM STYLES (CSS)
# ============================
st.markdown(
    """
    <style>
        html, body, .stApp {
            background-image: url('https://source.unsplash.com/1600x900/?data,technology');
            background-size: cover;
            background-attachment: fixed;
            color: white !important;
            font-family: 'Arial', sans-serif;
        }
        .st-emotion-cache-1kyxreq {
            background: rgba(28, 28, 28, 0.8) !important;
            border-radius: 10px;
            padding: 15px;
        }
        .stButton>button {
            background-color: #FF4500 !important;
            color: white !important;
            border-radius: 10px;
            font-weight: bold;
            padding: 8px 16px;
        }
        .stDataFrame, .dataframe tbody tr {
            background-color: #1C1C1C !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# 3. APP TITLE AND INTRODUCTION
# ============================
st.title("📊 Demand Forecasting App")
st.markdown("""
    ### Welcome to the Demand Forecasting App!
    Predict future demand using advanced statistical models.
    Simply provide past demand values and select the forecast horizon.
""")

# ============================
# 4. SIDEBAR FOR USER INPUTS
# ============================
st.sidebar.header("⚙️ User Input")
past_demand = st.sidebar.text_area("Enter past demand values (comma-separated):", "500,520,510,530,540")
days_to_forecast = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=365, value=30, step=1)

# ============================
# 5. DATA PROCESSING
# ============================
try:
    demand_values = list(map(float, past_demand.split(',')))
    dates = pd.date_range(start=pd.Timestamp.today() - pd.Timedelta(days=len(demand_values)), periods=len(demand_values), freq='D')
    data = pd.DataFrame({'Date': dates, 'Demand': demand_values}).set_index('Date')
except ValueError:
    st.error("Please enter valid numerical demand values separated by commas.")
    st.stop()

# ============================
# 6. TRAIN SARIMA MODEL
# ============================
st.spinner("Training SARIMA model...")
model = SARIMAX(data['Demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)

# ============================
# 7. FORECAST FUTURE DEMAND
# ============================
forecast = results.forecast(steps=days_to_forecast)
forecast_dates = pd.date_range(start=data.index[-1], periods=days_to_forecast + 1, freq='D')[1:]
forecast_series = pd.Series(forecast.values, index=forecast_dates)

# ============================
# 8. DISPLAY RESULTS
# ============================
st.markdown("### 📈 Forecasted Demand")
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines+markers', name='Forecasted Demand', line=dict(color='cyan', width=2)))
fig.update_layout(
    title='📈 Forecasted Demand',
    xaxis=dict(title='Date', showgrid=True, gridcolor='gray'),
    yaxis=dict(title='Predicted Demand', showgrid=True, gridcolor='gray'),
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    font=dict(color='white')
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📊 Historical vs Predicted Demand")
hist_fig = px.bar(x=data.index, y=data['Demand'], labels={'x': 'Date', 'y': 'Historical Demand'}, title='📊 Historical Demand', color_discrete_sequence=['#FFA07A'])
st.plotly_chart(hist_fig, use_container_width=True)

st.markdown("### 📋 Forecasted Demand Table")
st.dataframe(forecast_series.rename("Predicted Demand"), height=400)

# ============================
# 9. FOOTER
# ============================
st.markdown("---")
st.markdown("**Developed with ❤️ using Streamlit**", unsafe_allow_html=True)
