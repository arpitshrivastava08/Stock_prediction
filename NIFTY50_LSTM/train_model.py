# NIFTY50 Stock Price Prediction using Stacked LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Download NIFTY50 Data

ticker = "^NSEI"

df = yf.download(ticker, start="2010-01-01", end="2024-01-01")

print("Dataset downloaded successfully")
print(df.head())

# Select Close Price

df1 = df[['Close']]

plt.figure(figsize=(10,5))
plt.plot(df1)
plt.title("NIFTY50 Close Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.savefig("nifty50_price.png")
plt.show()

# Normalize Data

scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(df1)

# Train Test Split

training_size = int(len(df1)*0.65)

train_data = df1[0:training_size,:]
test_data = df1[training_size:len(df1),:]

# Create Dataset Function

def create_dataset(dataset, time_step=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])

    return np.array(dataX), np.array(dataY)

# Prepare Data

time_step = 100

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#  Build LSTM Model

model = Sequential()

model.add(LSTM(100, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(1))

model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

model.summary()

# Train Model

model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=100,
    batch_size=64,
    verbose=1
)

# Predictions

train_predict = model.predict(X_train)
test_predict  = model.predict(X_test)

# inverse transform
train_predict = scaler.inverse_transform(train_predict)
test_predict  = scaler.inverse_transform(test_predict)

y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# Calculate RMSE

rmse_train = np.sqrt(mean_squared_error(y_train_actual, train_predict))
rmse_test  = np.sqrt(mean_squared_error(y_test_actual, test_predict))

print("Train RMSE:", rmse_train)
print("Test RMSE:", rmse_test)

# Plot Predictions

look_back = 100

trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back] = train_predict

testPredictPlot = np.empty_like(df1)
testPredictPlot[:] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1] = test_predict

plt.figure(figsize=(12,6))

plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)

plt.legend(["Actual Price","Train Prediction","Test Prediction"])
plt.title("Stock Price Prediction using LSTM")

plt.savefig("prediction_graph.png")

plt.show()