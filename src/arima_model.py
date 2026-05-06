from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_arima(train_series):

    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    fitted_model = model.fit(disp=False)

    return fitted_model