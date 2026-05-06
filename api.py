from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load('models/best_model.pkl')


@app.get('/')
def home():

    return {
        'message': 'Forecasting API Running'
    }


@app.get('/predict')
def predict():

    try:

        # Dummy future data
        future_data = pd.DataFrame({
            'lag_1': [100] * 8,
            'lag_7': [100] * 8,
            'lag_30': [100] * 8,
            'rolling_mean_4': [100] * 8,
            'rolling_std_4': [1] * 8,
            'month': [1] * 8,
            'week': [1] * 8,
            'year': [2025] * 8,
            'day_of_week': [1] * 8,
            'is_holiday': [0] * 8
        })

        predictions = model.predict(future_data)

        return {
            'forecast_next_8_weeks': predictions.tolist()
        }

    except Exception as e:

        return {
            'error': str(e)
        }