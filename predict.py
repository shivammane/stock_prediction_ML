# Importing the libraries
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def hello(g):
    plt.subplots(figsize=(12, 4))
    dataset = pd.read_csv(f"static/{g}", index_col="Date", parse_dates=True)
    dataset["Close Price"] = dataset["Close Price"].replace(',', '').astype(float)
    dataset["Total Traded Quantity"] = dataset["Total Traded Quantity"].replace(',', '').astype(float)
    dataset.rolling(7).mean().head(20)
    dataset['Open Price'].plot(figsize=(16, 6))
    dataset.rolling(window=30).mean()['Close Price'].plot()
    dataset['Close: 30 Day Mean'] = dataset['Close Price'].rolling(window=30).mean()
    dataset[['Close Price', 'Close: 30 Day Mean']].plot(figsize=(16, 6))
    dataset['Close Price'].expanding(min_periods=1).mean().plot(figsize=(16, 6))
    training_set = dataset['Open Price']
    training_set = pd.DataFrame(training_set)
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    x_train = []
    y_train = []
    for i in range(30, dataset.shape[0]):
        x_train.append(training_set_scaled[i - 30:i, 0])
        y_train.append(training_set_scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshaping
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units=1))
    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(x_train, y_train, epochs=100, batch_size=32)
    dataset_test = pd.read_csv(f"static/{g}", index_col="Date", parse_dates=True)
    real_stock_price = dataset_test.iloc[:, 2:3].values
    dataset_test.head()
    os.remove(f"static/{g}")
    dataset_test["Total Traded Quantity"] = dataset_test["Total Traded Quantity"].replace(',', '').astype(float)
    pd.DataFrame(dataset_test['Open Price'])
    dataset_total = pd.concat((dataset['Open Price'], dataset_test['Open Price']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 30:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    x_test = []
    for i in range(30, dataset.shape[0] + 31):
        x_test.append(inputs[i - 30:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predicted_stock_price = regressor.predict(x_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    final_prediction = predicted_stock_price[-1][-1]
    predicted_stock_price = pd.DataFrame(predicted_stock_price)
    predicted_stock_price.info()
    plt.figure(figsize=(16, 6))
    plt.plot(real_stock_price, color='red', label='Real Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title(f'{g}')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.grid('minor')
    plt.legend()
    delete_images()
    plt.savefig(f"static/images/plotdata/{g}.png")
    return final_prediction


def delete_images():
    files = glob.glob(f'static/images/plotdata/*')
    for f in files:
        os.remove(f)
