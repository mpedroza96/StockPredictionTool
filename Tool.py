from datetime import datetime, timedelta
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import pandas_datareader
import tensorflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')

class Tool:
    # Variables
    predictionModel = []
    model = []
    trainingDataLen = 0
    data = []

    # Stock quote history
    def stockQuote(stockName):
        return web.DataReader(stockName, data_source='yahoo', start='2010-01-01', end=datetime.date(datetime.now()))

    # Show plot of stock prices
    def stockPlot(stock):
        plt.figure(figsize=(15, 8))
        plt.title('Price History')
        plt.plot(stock['Close'])
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Price USD', fontsize=20)
        plt.show()

    # Show plot of stock prices
    def stockTable(stock):
        table_data = []
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        for n in range(20):
            date = datetime.date(datetime.now() - timedelta(days = n))
            table_data.append([date, round(stock['Close'][-(n+1)], 2)])

        table = ax.table(cellText=table_data, loc='center', colLabels=("Date", "Closing Price"))
        table.set_fontsize(10)
        table.scale(1, 1.3)
        ax.axis('off')
        plt.show()

    # Convert and train data with LSTM
    def trainData(stock):
        data = stock.filter(['Close'])
        Tool.data = data
        dataset = data.values
        trainingDataLen = math.ceil(len(dataset) * 0.8)
        Tool.trainingDataLen = trainingDataLen

        # Scale Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaledData = scaler.fit_transform(dataset)
        trainData = scaledData[0:trainingDataLen]

        # Split Data
        x_train = []
        y_train = []

        for i in range(60, len(trainData)):
            x_train.append(trainData[i - 60:i, 0])
            y_train.append(trainData[i, 0])

        # Convert train data to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        Tool.model = model

        # Train model
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        # Create testing Data
        testData = scaledData[trainingDataLen - 60:, :]
        x_test = []
        y_test = dataset[trainingDataLen:, :]

        for i in range(60, len(testData)):
            x_test.append(testData[i - 60:i, 0])

        # Convert data to numpy array
        x_test = np.array(x_test)

        # Reshape Data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Get models predicted values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculate Error RMSE
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        print('RMSE for predictions: ', rmse)
        Tool.predictionModel = predictions

        return rmse

    def predictionPlot():
        if (Tool.data is None or Tool.predictionModel is None or Tool.trainingDataLen is None):
            print('Train model before plot')
        else:
            train = Tool.data[:Tool.trainingDataLen]
            valid = Tool.data[Tool.trainingDataLen:]
            valid['Predictions'] = Tool.predictionModel
            plt.figure(figsize=(15, 8))
            plt.title('Prediction Model')
            plt.xlabel('Date', fontsize=20)
            plt.ylabel('Price USD', fontsize=20)
            plt.plot(train['Close'])
            plt.plot(valid[['Close', 'Predictions']])
            plt.legend(['Data used for training model', 'Real Value', 'Predictions'], loc='lower right')
            plt.show()


    # Show plot of stock prices
    def predictionTable():
        if (Tool.data is None or Tool.predictionModel is None or Tool.trainingDataLen is None):
            print('Train model before plot')
        else:
            table_data = []
            fig = plt.figure(dpi=100)
            ax = fig.add_subplot(1, 1, 1)

            for n in range(20):
                date = datetime.date(datetime.now() - timedelta(days=n))
                table_data.append([date, Tool.predictionModel[-(n + 1)]])

            table = ax.table(cellText=table_data, loc='center', colLabels=("Date", "Closing Price"))
            table.set_fontsize(10)
            table.scale(1, 1.3)
            ax.axis('off')
            plt.show()


    def futurePrediction():
        lastDays = Tool.data[-60:].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        lastDaysScaled = scaler.fit_transform(lastDays)
        x_test = []
        x_test.append(lastDaysScaled)
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        pred = Tool.model.predict(x_test)
        pred = scaler.inverse_transform(pred)
        return pred

