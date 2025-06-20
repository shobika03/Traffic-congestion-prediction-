from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_lstm(X, y):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Scale features
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Reshape X for LSTM
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    model = Sequential()
    model.add(LSTM(50, input_shape=(1, X.shape[1])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_scaled, y_scaled, epochs=10, batch_size=64, verbose=0)

    return model, scaler_X, scaler_y
