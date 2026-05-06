import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

scaler = MinMaxScaler()

def create_sequences(data, seq_length=10):

    X = []
    y = []

    for i in range(len(data) - seq_length):

        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)


def train_lstm(series):

    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = create_sequences(scaled_data)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()

    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=20, verbose=0)

    return model, scaler