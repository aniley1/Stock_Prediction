# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load historical stock price data (replace 'AAPL' with the desired stock symbol)
import yfinance as yf
data = yf.download('AAPL', start='2010-01-01', end='2021-12-31')

# Use only the 'Close' prices for prediction
data = data['Close']

# Plot the stock price data
plt.figure(figsize=(12, 6))
plt.title('Stock Price History')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(data)
plt.show()

# Preprocess the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Create sequences for training and testing
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length]
        target = data[i+seq_length:i+seq_length+1]
        sequences.append((sequence, target))
    return sequences

seq_length = 10  # Adjust the sequence length as needed
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

# Convert sequences to numpy arrays
X_train, y_train = np.array([seq[0] for seq in train_sequences]), np.array([seq[1] for seq in train_sequences])
X_test, y_test = np.array([seq[0] for seq in test_sequences]), np.array([seq[1] for seq in test_sequences])

# Build an LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f'Training Loss: {train_loss:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the scaled data
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Plot the predictions vs. actual data
plt.figure(figsize=(12, 6))
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(data.index[seq_length:train_size], train_predictions, label='Train Predictions', color='blue')
plt.plot(data.index[train_size+seq_length:], test_predictions, label='Test Predictions', color='red')
plt.plot(data.index[seq_length:train_size], data.values[seq_length:train_size], label='Actual Train Data', color='green')
plt.plot(data.index[train_size+seq_length:], data.values[train_size+seq_length:], label='Actual Test Data', color='purple')
plt.legend()
plt.show()
