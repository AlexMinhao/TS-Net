import os

import numpy as np
import pandas as pd


from datetime import datetime

import numpy
import math
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D











if __name__ == '__main__':



    data = pd.read_csv('F:/school/Papers/timeseriesNew/TS-Net/dataset/SP500/individual_stocks_5yr/individual_stocks_5yr//GOOG_data.csv')
    print(data.head())

    dataset = data.iloc[:,1].values
    dataset = dataset.reshape(-1,1) # (975,) sometimes can be problem
    dataset = dataset.astype("float32")
    print(dataset.shape)

    # scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # train test split
    train_size = int(len(dataset) * 0.75) # Split dataset 75% for train set, 25% for test set
    test_size = len(dataset) - train_size
    train = dataset[0:train_size,:]
    test = dataset[train_size:len(dataset),:]
    print("train size: {}, test size: {} ".format(len(train), len(test)))

    time_stemp = 10
    dataX = []
    dataY = []
    for i in range(len(train)-time_stemp-1):
        a = train[i:(i+time_stemp), 0]
        dataX.append(a)
        dataY.append(train[i + time_stemp, 0])
    trainX = numpy.array(dataX)
    trainY = numpy.array(dataY)

    dataX = []
    dataY = []
    for i in range(len(test)-time_stemp-1):
        a = test[i:(i+time_stemp), 0]
        dataX.append(a)
        dataY.append(test[i + time_stemp, 0])
    testX = numpy.array(dataX)
    testY = numpy.array(dataY)

    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# Vanilla LSTM
# model
# A Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM units, and an output layer used to make a prediction.
#     model = Sequential()
#     model.add(LSTM(10, input_shape=(1, time_stemp))) # 10 lstm neuron(block)
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     model.fit(trainX, trainY, epochs=50, batch_size=1)
#
#     trainPredict = model.predict(trainX)
#     testPredict = model.predict(testX)
#     # invert predictions
#     trainPredict = scaler.inverse_transform(trainPredict)
#     trainY = scaler.inverse_transform([trainY])
#     testPredict = scaler.inverse_transform(testPredict)
#     testY = scaler.inverse_transform([testY])
#     # calculate root mean squared error
#     trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#     print('Train Score: %.2f RMSE' % (trainScore))
#     testScore_vanilla = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#     print('Test Score: %.2f RMSE' % (testScore_vanilla))

    # # shifting train
    # trainPredictPlot = numpy.empty_like(dataset)
    # trainPredictPlot[:, :] = numpy.nan
    # trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
    # # shifting test predictions for plotting
    # testPredictPlot = numpy.empty_like(dataset)
    # testPredictPlot[:, :] = numpy.nan
    # testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict
    # # plot baseline and predictions
    # f,ax = plt.subplots(figsize = (30,7))
    # plt.plot(scaler.inverse_transform(dataset))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.show()
    #
    #########################################################################################
    # 2 - Stacked LSTM
    #Multiple hidden LSTM layers can be stacked one on top of another in what is referred to as a Stacked LSTM model.
    # define model
    # model = Sequential()
    # model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1,time_stemp)))
    # model.add(LSTM(50, activation='relu'))
    # model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mse')
    # model.fit(trainX, trainY, epochs=50, batch_size=1)
    #
    # trainPredict = model.predict(trainX)
    # testPredict = model.predict(testX)
    # # invert predictions
    # trainPredict = scaler.inverse_transform(trainPredict)
    # trainY = scaler.inverse_transform([trainY])
    # testPredict = scaler.inverse_transform(testPredict)
    # testY = scaler.inverse_transform([testY])
    # # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore_Stacked = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    # print('Test Score: %.2f RMSE' % (testScore_Stacked))

# 3 - Bidirectional LSTM
#On some sequence prediction problems, it can be beneficial to allow the LSTM model to learn the input sequence both forward and backwards and concatenate both interpretations.
    # define model
    # model = Sequential()
    # model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(1, time_stemp)))
    # model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mse')
    # model.fit(trainX, trainY, epochs=50, batch_size=1)
    #
    # trainPredict = model.predict(trainX)
    # testPredict = model.predict(testX)
    # # invert predictions
    # trainPredict = scaler.inverse_transform(trainPredict)
    # trainY = scaler.inverse_transform([trainY])
    # testPredict = scaler.inverse_transform(testPredict)
    # testY = scaler.inverse_transform([testY])
    # # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore_bidirectional = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore_bidirectional))

#4 - CNN LSTM
# A CNN model can be used in a hybrid model with an LSTM backend where the CNN is used to interpret subsequences of
# input that together are provided as a sequence to an LSTM model to interpret.

    dataset = data.iloc[:, 1:6].values
    dataset = dataset.reshape(-1, 5)  # (975,) sometimes can be problem
    dataset = dataset.astype("float32")
    # scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # train test split
    train_size = int(len(dataset) * 0.75)  # Split dataset 75% for train set, 25% for test set
    test_size = len(dataset) - train_size
    train = dataset[0:train_size, :] #(731, 5)
    test = dataset[train_size:len(dataset), :] #(244, 5)
    time_stemp = 7
    horizon = 1#,3,5
    dataX = []
    dataY = []
    for i in range(len(train) - time_stemp - horizon):
        a = train[i:(i + time_stemp), :]
        dataX.append(a)
        dataY.append(train[i + time_stemp:i + time_stemp+horizon, :])
    trainX = numpy.array(dataX)
    trainY = numpy.array(dataY)
    dataX = []
    dataY = []
    for i in range(len(test) - time_stemp -horizon):
        a = test[i:(i + time_stemp), :]
        dataX.append(a)
        dataY.append(test[i + time_stemp:i + time_stemp+horizon, :])
    testX = numpy.array(dataX)
    testY = numpy.array(dataY)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, 1, testX.shape[1]))



    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 1, time_stemp)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, padding='same')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(trainX, trainY, epochs=50, batch_size=1)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore_cnn = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore_cnn))