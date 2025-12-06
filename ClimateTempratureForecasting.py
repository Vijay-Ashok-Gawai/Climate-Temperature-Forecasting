# %%

import os

# 1. Define a safe project folder path inside Documents
folder_path = r"C:\Users\Dell\Documents\project"

# 2. Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# 3. Change current working directory to the project folder
%cd C:/Users/Dell/Documents/project

# 4. Confirm
print("Current working directory:", os.getcwd())


import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler



# Statsmodels SARIMA
import statsmodels.api as sm

# Prophet 
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None


# TensorFlow / Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:
    tf = None


# -----------------------------
# 1) Data loading & basic checks
# -----------------------------

def load_data(path=r'C:\\Users\\Dell\\Documents\\project\\data\\temperature.csv', date_col='date', value_col='temp'):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df[[date_col, value_col]].rename(columns={date_col: 'date', value_col: 'meantemp'})
    df = df.set_index('date')
    return df


# -----------------------------
# 2) Exploratory Data Analysis
# -----------------------------

def eda(df, resample_rule='D'):
    """Simple EDA: resample, plot, rolling stats"""
    ts = df['meantemp'].resample(resample_rule).mean()

    print('\n-- Basic stats --')
    print(ts.describe())

    plt.figure(figsize=(12,4))
    ts.plot(title='Temperature time series')
    plt.ylabel('Temperature')
    plt.show()

    # Seasonal decomposition
    try:
        decomposition = sm.tsa.seasonal_decompose(ts.dropna(), model='additive', period=365)
        decomposition.plot()
        plt.suptitle('Seasonal decomposition (additive)')
        plt.show()
    except Exception as e:
        print('Seasonal decomposition failed:', e)

    # Autocorrelation
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(121)
    sm.graphics.tsa.plot_acf(ts.dropna(), ax=ax1, lags=50)
    ax2 = fig.add_subplot(122)
    sm.graphics.tsa.plot_pacf(ts.dropna(), ax=ax2, lags=50)
    plt.show()

    return ts


# -----------------------------
# 3) Train / test split (time-based)
# -----------------------------

def train_test_split_series(ts, test_periods=365):
    """Split series into train/test by last `test_periods` time steps"""
    ts = ts.dropna()
    train = ts.iloc[:-test_periods]
    test = ts.iloc[-test_periods:]
    return train, test


# -----------------------------
# 4) Baseline model: naive persistence
# -----------------------------

def persistence_forecast(train, test):
    """Naive forecast: forecast(t) = value(t-1)"""
    preds = train.iloc[-1:].repeat(len(test)).values.flatten()
    preds = pd.Series(preds, index=test.index)
    return preds


# -----------------------------
# 5) SARIMA (statsmodels)
# -----------------------------

def train_sarima(train, order=(1,1,1), seasonal_order=(1,1,1,12), steps=365):
    """Train SARIMA and forecast `steps` ahead"""
    model = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                      enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    forecast = res.get_forecast(steps=steps)
    fc = forecast.predicted_mean
    conf = forecast.conf_int()
    return res, fc, conf


# -----------------------------
# 6) Prophet model
# -----------------------------

def train_prophet(train, periods=365, freq='D'):
    if Prophet is None:
        raise ImportError('Prophet is not installed. pip install prophet')
    df = train.reset_index().rename(columns={'date':'ds', 'temp':'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    fc = forecast.set_index('ds')['yhat'][-periods:]
    conf_low = forecast.set_index('ds')['yhat_lower'][-periods:]
    conf_high = forecast.set_index('ds')['yhat_upper'][-periods:]
    conf = pd.concat([conf_low, conf_high], axis=1)
    return m, fc, conf


# -----------------------------
# 7) LSTM (deep learning)
# -----------------------------

def create_lstm_dataset(series, look_back=30):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i+look_back)])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)


def train_lstm(train, test, look_back=30, epochs=20, batch_size=16):
    if tf is None:
        raise ImportError('TensorFlow is not installed. pip install tensorflow')

    scaler = MinMaxScaler()
    all_vals = np.concatenate([train.values.reshape(-1,1), test.values.reshape(-1,1)], axis=0)
    scaler.fit(all_vals)

    train_scaled = scaler.transform(train.values.reshape(-1,1)).flatten()
    test_scaled = scaler.transform(test.values.reshape(-1,1)).flatten()

    X_train, y_train = create_lstm_dataset(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, input_shape=(look_back,1), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Forecast rolling over test set
    history = train_scaled[-look_back:].tolist()
    preds = []
    for t in range(len(test_scaled)):
        x_input = np.array(history[-look_back:]).reshape((1, look_back, 1))
        yhat = model.predict(x_input, verbose=0)[0,0]
        preds.append(yhat)
        history.append(test_scaled[t]) 

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    preds = pd.Series(preds, index=test.index)
    return model, preds


# -----------------------------
# 8) Evaluation helpers
# -----------------------------

def evaluate_forecast(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mae': mae}


# -----------------------------
# 9) Plot forecasts
# -----------------------------

def plot_forecasts(train, test, predictions_dict, title='Forecast comparison'):
    plt.figure(figsize=(14,6))
    train.plot(label='Train')
    test.plot(label='Test', color='black')

    for name, series in predictions_dict.items():
        series.plot(label=name)

    plt.legend()
    plt.title(title)
    plt.show()


# -----------------------------
# 10) Example run (main)
# -----------------------------
if __name__ == '__main__':
    # === USER CONFIG ===
    DATA_PATH = r'C:\\Users\\Dell\\Documents\\project\\data\\temperature.csv'  # change to your file
    DATE_COL = 'date'
    VALUE_COL = 'meantemp'
    TEST_DAYS = 365  # last 365 days as test
    LOOK_BACK = 30
    LSTM_EPOCHS = 10

    # Load
    try:
        df = load_data(DATA_PATH, DATE_COL, VALUE_COL)
    except Exception as e:
        print('Error loading data:', e)
        print('\nPlease place your CSV at', DATA_PATH, 'with columns', DATE_COL, ',', VALUE_COL)
        raise SystemExit

    # EDA
    ts = eda(df, resample_rule='D')

    # Train / test split
    train, test = train_test_split_series(ts, test_periods=TEST_DAYS)
    print('\nTrain length:', len(train), 'Test length:', len(test))

    # Persistence baseline
    pers_preds = persistence_forecast(train, test)
    print('\nPersistence eval:', evaluate_forecast(test.values, pers_preds.values))

    # SARIMA
    print('\nTraining SARIMA (this may take time)...')
    try:
        sarima_res, sarima_fc, sarima_conf = train_sarima(train, order=(2,1,2), seasonal_order=(1,1,1,12), steps=len(test))
        sarima_fc = sarima_fc.reindex(test.index).fillna(method='ffill')
        print('SARIMA eval:', evaluate_forecast(test.values, sarima_fc.values))
    except Exception as e:
        print('SARIMA failed:', e)
        sarima_fc = pd.Series(np.nan, index=test.index)

    # Prophet
    if Prophet is not None:
        try:
            print('\nTraining Prophet...')
            prophet_model, prophet_fc, prophet_conf = train_prophet(train, periods=len(test), freq='D')
            prophet_fc = prophet_fc.reindex(test.index)
            print('Prophet eval:', evaluate_forecast(test.values, prophet_fc.values))
        except Exception as e:
            print('Prophet failed:', e)
            prophet_fc = pd.Series(np.nan, index=test.index)
    else:
        print('\nProphet not installed — skipping')
        prophet_fc = pd.Series(np.nan, index=test.index)

    # LSTM
    try:
        print('\nTraining LSTM (this may take time and requires TensorFlow)...')
        lstm_model, lstm_preds = train_lstm(train, test, look_back=LOOK_BACK, epochs=LSTM_EPOCHS, batch_size=16)
        print('LSTM eval:', evaluate_forecast(test.values, lstm_preds.values))
    except Exception as e:
        print('LSTM failed:', e)
        lstm_preds = pd.Series(np.nan, index=test.index)

    # Plot
    preds = {
        'Persistence': pers_preds,
        'SARIMA': sarima_fc,
        'Prophet': prophet_fc,
        'LSTM': lstm_preds
    }
    plot_forecasts(train[-(TEST_DAYS*2):], test, preds, title='Temperature forecasting comparison')

    # Final metrics
    print('\nFinal evaluation metrics:')
    for name, series in preds.items():
        try:
            metrics = evaluate_forecast(test.values, series.values)
            print(f"{name}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")
        except Exception:
            print(f"{name}: No predictions")

    print('\nDone. Adjust hyperparameters or try more advanced feature engineering (exogenous vars, climate indices).')

# %%
