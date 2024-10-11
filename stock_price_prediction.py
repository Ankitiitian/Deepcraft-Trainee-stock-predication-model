# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import math

# 1. Data Understanding and EDA

# Load the stock data
data = pd.read_csv('stock_price.csv')  # Replace with the actual file path

# Convert 'Date' column to datetime if it's not already
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Sort data by date in ascending order
data.sort_values('Date', inplace=True)

# Set 'Date' as the index for easier plotting
data.set_index('Date', inplace=True)

# Display the first few rows of the data
print("First few rows of the dataset:")
print(data.head())

# Get basic statistics
print("\nBasic statistical details:")
print(data.describe())

# Plot the closing prices over time
plt.figure(figsize=(14, 7))
plt.plot(data['Close Price'], label='Closing Price')
plt.title('NTT Stock Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (JPY)')
plt.legend()
plt.show()

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

for column in ['Volume']:  # Add other columns if needed
    data[column] = data[column].str.replace('B', 'e9').str.replace('M', 'e6').astype(float)


for column in data.select_dtypes(include=['object']).columns:
    # Check if the column contains percentage-like strings
    if data[column].str.contains('%').any():
        # Remove '%' and convert to float
        data[column] = data[column].str.rstrip('%').astype(float) / 100


# Correlation matrix for the features
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# 2. Data Preprocessing and Feature Engineering

# Handle missing values (if any)
# For simplicity, we'll forward-fill missing values
data.fillna(method='ffill', inplace=True)

# Ensure no remaining missing values
data.dropna(inplace=True)

# Create additional features (e.g., moving averages)
data['MA20'] = data['Close Price'].rolling(window=20).mean()
data['MA50'] = data['Close Price'].rolling(window=50).mean()

# Create the target variable (next day's closing price)
data['Future_Price'] = data['Close Price'].shift(-1)

# Drop the last row as it will have NaN in 'Future_Price'
data.dropna(inplace=True)

# Visualize the moving averages
plt.figure(figsize=(14, 7))
plt.plot(data['Close Price'], label='Closing Price')
plt.plot(data['MA20'], label='20-Day MA')
plt.plot(data['MA50'], label='50-Day MA')
plt.title('Closing Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (JPY)')
plt.legend()
plt.show()

# Feature selection
features = ['Close Price', 'MA20', 'MA50']

# Scale the features using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[features])

# Scale the target variable separately
scaler_target = MinMaxScaler(feature_range=(0, 1))
scaled_target = scaler_target.fit_transform(data[['Future_Price']])

# Split the data into training and test sets (80% train, 20% test)
train_size = int(len(data) * 0.8)

train_features = scaled_features[:train_size]
train_target = scaled_target[:train_size]

test_features = scaled_features[train_size:]
test_target = scaled_target[train_size:]

# 3. Model Selection and Training

# Prepare the data for LSTM (create sequences)
def create_sequences(features, target, time_steps=50):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i + time_steps])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 50  # You can adjust this value

X_train, y_train = create_sequences(train_features, train_target, time_steps)
X_test, y_test = create_sequences(test_features, test_target, time_steps)

# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,  # You can increase this number for better results
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Plot training & validation loss values
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 4. Model Evaluation and Results Analysis

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values
predicted_prices = scaler_target.inverse_transform(predictions)
actual_prices = scaler_target.inverse_transform(y_test)

# Calculate evaluation metrics
rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Plot actual vs predicted stock prices
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('NTT Stock Price Prediction - Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Price (JPY)')
plt.legend()
plt.show()

# 5. Consideration of Improvement Measures and Retraining the Model

# Possible improvements:
# - Increase the number of epochs
# - Adjust the time_steps
# - Modify the number of LSTM units
# - Add more features (e.g., technical indicators)
# - Try different activation functions or optimizers

# Example: Retraining the model with more epochs and additional features

# Add more technical indicators as features (e.g., RSI, MACD)
# For simplicity, we'll add only one more feature here

# Compute Relative Strength Index (RSI)
delta = data['Close Price'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
window_length = 14
roll_up = up.rolling(window=window_length).mean()
roll_down = down.rolling(window=window_length).mean()
RS = roll_up / roll_down
RSI = 100.0 - (100.0 / (1.0 + RS))
data['RSI'] = RSI

# Update the features list
features = ['Close Price', 'MA20', 'MA50', 'RSI']

# Drop NaN values resulted from RSI calculation
data.dropna(inplace=True)

# Rescale the features and target
scaled_features = scaler.fit_transform(data[features])
scaled_target = scaler_target.fit_transform(data[['Future_Price']])

# Split the data again
train_size = int(len(data) * 0.8)
train_features = scaled_features[:train_size]
train_target = scaled_target[:train_size]
test_features = scaled_features[train_size:]
test_target = scaled_target[train_size:]

# Create sequences
X_train, y_train = create_sequences(train_features, train_target, time_steps)
X_test, y_test = create_sequences(test_features, test_target, time_steps)

# Rebuild the model with adjusted parameters
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Retrain the model with more epochs
history = model.fit(
    X_train, y_train,
    epochs=150,  # Increased epochs
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Plot training & validation loss values
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Retraining')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values
predicted_prices = scaler_target.inverse_transform(predictions)
actual_prices = scaler_target.inverse_transform(y_test)

# Calculate evaluation metrics
rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)

print(f'After Improvement - Root Mean Squared Error (RMSE): {rmse}')
print(f'After Improvement - Mean Absolute Error (MAE): {mae}')

# Plot actual vs predicted stock prices after retraining
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('NTT Stock Price Prediction After Improvement - Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Price (JPY)')
plt.legend()
plt.show()
