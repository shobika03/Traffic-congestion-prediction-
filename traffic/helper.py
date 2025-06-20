import pandas as pd

def preprocess_data(path):
    df = pd.read_csv(path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month
    df['dayofweek'] = df['date_time'].dt.dayofweek

    features = ['hour', 'day', 'month', 'dayofweek', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']
    target = 'traffic_volume'

    X = df[features]
    y = df[target]
    return X, y, df
