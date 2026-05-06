# End-to-End Time Series Forecasting System

## Overview

This project forecasts the next 8 weeks of sales for each state using historical sales data.  
It trains multiple forecasting models, compares their performance, selects the best model automatically, and exposes predictions through a REST API.

-----

# Features

- Data preprocessing
- Missing value handling
- Missing date handling
- Feature engineering
- Multiple forecasting models
- Automatic model selection
- REST API for predictions

-----

# Models Implemented

- ARIMA / SARIMA
- Facebook Prophet
- XGBoost
- LSTM

-----

# Feature Engineering

- Lag Features (`lag_1`, `lag_7`, `lag_30`)
- Rolling Mean & Standard Deviation
- Month, Week, Year, Day Features
- Holiday Flag
- Time-Series Train/Validation Split

-----

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas, NumPy | Data Processing |
| Statsmodels | ARIMA/SARIMA |
| Prophet | Forecasting |
| XGBoost | ML Model |
| TensorFlow/Keras | LSTM |
| FastAPI | REST API |

-----

# Project Structure

```text
forecasting_project/
│
├── data/
├── models/
├── src/
├── train.py
├── api.py
├── requirements.txt
└── README.md
```

-----

# Installation

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment (Windows)

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

-----

# Run Training

```bash
python train.py
```

This will:
- Load dataset
- Train all models
- Compare performance
- Save best model

-----

# Run API

```bash
uvicorn api:app --reload
```

API runs at:

```text
http://127.0.0.1:8000
```

-----

# API Endpoints

## Home

```text
/
```

## Prediction

```text
/predict
```

Returns next 8-week forecast.

-----

# Evaluation Metrics

- MAE
- RMSE
- MAPE

Best model is selected using lowest RMSE.

-----

# Assignment Requirements Covered

- Multiple forecasting algorithms
- Missing value handling
- Feature engineering
- Seasonality & trend handling
- Time-series validation split
- Automatic model selection
- REST API deployment

-----

# Future Improvements

- Hyperparameter tuning
- Docker deployment
- Cloud deployment
- Dashboard integration

-----

# Author
Praveen Kumar Billingi
CSE - AI & ML Student