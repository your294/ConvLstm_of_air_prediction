from cmath import sqrt

import tensorflow as tf
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Add, TimeDistributed, MaxPooling2D, concatenate, \
    MaxPooling1D, Conv1D, LSTM, Flatten
from keras.layers import ConvLSTM2D, Conv2D, Dropout, UpSampling2D

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # 预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 拼接 put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除值为NAN的行 drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_acc15(result, predict):
    length = result.shape[0]
    cnt = 0
    for i in range(1, length, 1):
        if abs(result[i] - predict[i]) < result[i] * 0.30:
            cnt = cnt + 1
    return cnt / length


dataset = pd.read_csv("data/pollution.csv", header=0, index_col=0)
values = dataset.values

# values = tf.convert_to_tensor(values)
encoder = LabelEncoder()
pre = values
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
reframed = reframed.values

train_hours = 365 * 24 * 4
train = reframed[:train_hours, :]
test = reframed[train_hours:, :]
data_x_train = train[:, :-1]
data_y_train = train[:, -1]
data_x_test = test[:, :-1]
data_y_test = test[:, -1]
# 构建3D输入数据
data_x_train = data_x_train.reshape(data_x_train.shape[0], 1, data_x_train.shape[1])
data_x_test = data_x_test.reshape(data_x_test.shape[0], 1, data_x_test.shape[1])
# 构建模型
with tf.device("/gpu:0"):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', strides=1,
                     activation='relu',
                     input_shape=(1, data_x_train.shape[2])))  # input_shape=(X_train.shape[1], X_train.shape[2])
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', strides=1,
                     activation='relu',
                     input_shape=(1, data_x_train.shape[2])))
    model.add(MaxPooling1D(pool_size=1))
    model.add(LSTM(200, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
    model.add(LSTM(100, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(data_x_train, data_y_train, epochs=10, batch_size=32, verbose=2,
                        validation_data=(data_x_test, data_y_test), shuffle=False)
    # 输出 plot history
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()

    yhat = model.predict(data_x_test)
    data_x_test = data_x_test.reshape(data_x_test.shape[0], data_x_test.shape[2])
    yhat = np.concatenate((yhat, data_x_test[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, 0]
    inv_yhat = np.array(inv_yhat)

    data_y_test = data_y_test.reshape(len(data_y_test), 1)
    test_y = np.concatenate((data_y_test, data_x_test[:, 1:]), axis=1)
    inv_y_test = scaler.inverse_transform(test_y)
    inv_y_test = inv_y_test[:, 0]

    x = range(0, len(inv_yhat))
    pyplot.plot(x, inv_yhat, label="prediction", color="blue")
    pyplot.plot(x, inv_y_test, label="true", color="red")
    pyplot.legend()
    pyplot.show()

    acc15 = get_acc15(inv_y_test, inv_yhat)
    print(f'acc15 = {acc15}')
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y_test, inv_yhat))
    print(f'Test RMSE: {rmse}')
