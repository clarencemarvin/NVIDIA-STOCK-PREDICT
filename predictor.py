import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_fetcher import fetch_data

# Fetch NVDA data
start_date = '2010-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
df = fetch_data('NVDA', start_date, end_date)

if df is None:
    print("Failed to fetch data. Exiting.")
    exit()

# Rename columns to match the previous code
df = df.rename(columns={
    'open': 'Open Price',
    'high': 'High Price',
    'low': 'Low Price',
    'close': 'Close Price',
    'adjClose': 'Adj Close Price',
    'volume': 'Volume'
})

# RSI calculation
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# MACD calculation
def calculate_macd(prices, fast_period=12, slow_period=26):
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    return fast_ema - slow_ema

# Function to add features
def add_features(df):
    df['MA10'] = df['Adj Close Price'].rolling(window=10).mean()
    df['MA30'] = df['Adj Close Price'].rolling(window=30).mean()
    df['RSI'] = calculate_rsi(df['Adj Close Price'])
    df['MACD'] = calculate_macd(df['Adj Close Price'])
    return df

# Add features
nvda = add_features(df)

# Create lags
for i in range(1, 4):
    nvda[f'Lag_{i}'] = nvda['Adj Close Price'].shift(i)

nvda = nvda.dropna()

# Define features and target
features = ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume', 'MA10', 'MA30', 'RSI', 'MACD', 'Lag_1', 'Lag_2', 'Lag_3']
target = 'Adj Close Price'

X = nvda[features]
y = nvda[target]

# Feature selection
selector = SelectKBest(score_func=f_regression, k=8)
X_selected = selector.fit_transform(X, y)
selected_features = [features[i] for i in selector.get_support(indices=True)]

# Split the data
train_size = int(len(X_selected) * 0.8)
X_train, X_test = X_selected[:train_size], X_selected[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale the features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Hyperparameter tuning for XGBoost
xgb_params = {
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0]
}
xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid.fit(X_train_scaled, y_train_scaled)
xgb_model = xgb_grid.best_estimator_

# Implement time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
xgb_scores = []

for train_index, val_index in tscv.split(X_train_scaled):
    X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_cv, y_val_cv = y_train_scaled[train_index], y_train_scaled[val_index]
    
    xgb_model.fit(X_train_cv, y_train_cv)
    xgb_pred_cv = xgb_model.predict(X_val_cv)
    xgb_scores.append(mean_squared_error(y_val_cv, xgb_pred_cv))

print(f"XGBoost CV MSE: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")

# Fit XGBoost on the entire training set
xgb_model.fit(X_train_scaled, y_train_scaled)

# LSTM model
input_layer = Input(shape=(1, X_train_scaled.shape[1]))
lstm_layer = LSTM(units=64, activation='relu')(input_layer)
output_layer = Dense(units=1)(lstm_layer)

lstm_model = Model(inputs=input_layer, outputs=output_layer)

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

lstm_model.fit(
    X_train_lstm, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Make predictions
xgb_pred = scaler_y.inverse_transform(xgb_model.predict(X_test_scaled).reshape(-1, 1))
lstm_pred = scaler_y.inverse_transform(lstm_model.predict(X_test_lstm))

# Ensemble predictions with different weights
ensemble_weights = [0.5, 0.5]  # Equal weights for XGBoost and LSTM
ensemble_pred = ensemble_weights[0] * xgb_pred + ensemble_weights[1] * lstm_pred

# Moving Average (MA) prediction
ma_window = 10
ma_pred = nvda['Adj Close Price'].rolling(window=ma_window).mean().shift(1)
ma_pred_test = ma_pred[train_size:]

# Combine ensemble and MA predictions
combined_pred = 0.7 * ensemble_pred + 0.3 * ma_pred_test.values.reshape(-1, 1)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

evaluate_model(y_test, xgb_pred, "XGBoost")
evaluate_model(y_test, lstm_pred, "LSTM")
evaluate_model(y_test, ensemble_pred, "Weighted Ensemble")
evaluate_model(y_test, combined_pred, "Combined Ensemble and MA")

# Calculate the direction of actual price changes
actual_direction = np.sign(y_test.diff().dropna())

# Calculate the direction of predicted price changes
predicted_direction = np.sign(np.diff(combined_pred.flatten()))

# Ensure both arrays have the same length
min_length = min(len(actual_direction), len(predicted_direction))
actual_direction = actual_direction[-min_length:]
predicted_direction = predicted_direction[-min_length:]
test_accuracy = np.mean(actual_direction == predicted_direction)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Calculate training accuracy
train_xgb_pred = scaler_y.inverse_transform(xgb_model.predict(X_train_scaled).reshape(-1, 1))
train_lstm_pred = scaler_y.inverse_transform(lstm_model.predict(X_train_lstm))
train_ensemble_pred = ensemble_weights[0] * train_xgb_pred + ensemble_weights[1] * train_lstm_pred
train_ma_pred = ma_pred[:train_size]
train_combined_pred = 0.7 * train_ensemble_pred + 0.3 * train_ma_pred.values.reshape(-1, 1)

train_actual_direction = np.sign(y_train.diff().dropna())
train_predicted_direction = np.sign(np.diff(train_combined_pred.flatten()))

train_min_length = min(len(train_actual_direction), len(train_predicted_direction))
train_actual_direction = train_actual_direction[-train_min_length:]
train_predicted_direction = train_predicted_direction[-train_min_length:]

train_accuracy = np.mean(train_actual_direction == train_predicted_direction)
print(f"Training Set Accuracy: {train_accuracy:.4f}")

historical_dates = nvda.index[train_size:]
historical_predictions = pd.Series(ensemble_pred.flatten(), index=historical_dates)
ma_predictions = pd.Series(ma_pred_test, index=historical_dates)
combined_predictions = pd.Series(combined_pred.flatten(), index=historical_dates)

# Plot results
plt.figure(figsize=(15, 8))
plt.plot(nvda.index, nvda['Adj Close Price'], label='Actual Price', alpha=0.7)
plt.plot(historical_dates, combined_predictions, label='Combined Ensemble and MA Prediction', alpha=0.7)

plt.title('NVIDIA Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Adj Close Price USD($)')
plt.legend()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()
