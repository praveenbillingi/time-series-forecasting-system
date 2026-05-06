import os
import joblib
import pandas as pd

from src.data_preprocessing import load_data
from src.data_preprocessing import preprocess_data
from src.data_preprocessing import create_features

from src.train_test_split import split_data
from src.evaluation import evaluate_model
from src.visualization import plot_forecast

from src.arima_model import train_arima
from src.prophet_model import train_prophet
from src.xgboost_model import train_xgboost
from src.xgboost_model import FEATURES
from src.lstm_model import train_lstm


DATA_PATH = 'data/sales_data.xlsx'


def main():

    print('Loading data...')

    df = load_data(DATA_PATH)

    df = preprocess_data(df)

    df = create_features(df)

    all_results = []

    best_model_name = None
    best_score = float('inf')
    best_model = None

    first_state = df['State'].unique()[0]

    state_df = df[df['State'] == first_state]

    train_df, val_df = split_data(state_df)

    print('Training ARIMA...')

    arima_model = train_arima(train_df['Total'])

    arima_preds = arima_model.forecast(len(val_df))

    arima_metrics = evaluate_model(val_df['Total'], arima_preds)

    all_results.append(['ARIMA', arima_metrics])

    print('Training Prophet...')

    prophet_model = train_prophet(train_df)

    future = prophet_model.make_future_dataframe(periods=len(val_df), freq='W')

    forecast = prophet_model.predict(future)

    prophet_preds = forecast['yhat'].tail(len(val_df))

    prophet_metrics = evaluate_model(val_df['Total'], prophet_preds)

    all_results.append(['Prophet', prophet_metrics])

    print('Training XGBoost...')

    xgb_model = train_xgboost(train_df)

    xgb_preds = xgb_model.predict(val_df[FEATURES])

    xgb_metrics = evaluate_model(val_df['Total'], xgb_preds)

    all_results.append(['XGBoost', xgb_metrics])

    print('Training LSTM...')

    lstm_model, scaler = train_lstm(train_df['Total'])

    print('\nMODEL RESULTS\n')

    for model_name, metrics in all_results:

        print(model_name)
        print(metrics)
        print()

        if metrics['RMSE'] < best_score:
            best_score = metrics['RMSE']
            best_model_name = model_name

    print(f'Best Model: {best_model_name}')

    # Save best model
    if best_model_name == 'ARIMA':
        joblib.dump(arima_model, 'models/best_model.pkl')

    elif best_model_name == 'Prophet':
        joblib.dump(prophet_model, 'models/best_model.pkl')

    elif best_model_name == 'XGBoost':
        joblib.dump(xgb_model, 'models/best_model.pkl')

    print('Best model saved successfully.')

    results_df = pd.DataFrame([
    {
        'Model': model_name,
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'MAPE': metrics['MAPE']
    }
    for model_name, metrics in all_results
    ])

    results_df.to_csv('outputs/model_results.csv', index=False)

if __name__ == '__main__':
    main()