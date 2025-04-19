from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import io
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/forecast/")
async def forecast(file: UploadFile = File(...), periods: int = 10):
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Ensure the data has a time column and a sales column
        df.columns = df.columns.str.lower()
        if 'date' not in df.columns or 'sales' not in df.columns:
            return {"error": "CSV must contain 'Date' and 'Sales' columns."}

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Apply Holt-Winters Exponential Smoothing for forecasting
        model = ExponentialSmoothing(df['sales'], trend="add", seasonal="add", seasonal_periods=12)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(periods)

        # Convert forecast to JSON response
        forecast_data = {str(df.index[-1] + pd.DateOffset(days=i+1)): float(val) for i, val in enumerate(forecast)}

        return {"forecast": forecast_data}

    except Exception as e:
        return {"error": str(e)}
