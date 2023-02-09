import pandas as pd
import numpy as np
from matplotlib import pyplot
import os

print(os.getcwd())
san_men = pd.read_csv('../he_nan_data/clean_pollute2/san_men_clean.csv')
xin_yang = pd.read_csv('../he_nan_data/clean_pollute2/xin_yang_clean.csv')
nan_yang = pd.read_csv('../he_nan_data/clean_pollute2/nan_yang_clean.csv')
zhou_kou = pd.read_csv('../he_nan_data/clean_pollute2/zhou_kou_clean.csv')
shang_qiu = pd.read_csv('../he_nan_data/clean_pollute2/shang_qiu_clean.csv')

san_men = san_men[['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'temp', 'humi', 'pressure']]
xin_yang = xin_yang[['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'temp', 'humi', 'pressure']]
nan_yang = nan_yang[['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'temp', 'humi', 'pressure']]
zhou_kou = zhou_kou[['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'temp', 'humi', 'pressure']]
shang_qiu = shang_qiu[['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'temp', 'humi', 'pressure']]

col_san_men = ['aqi_s', 'pm2_5_s', 'pm10_s', 'so2_s', 'no2_s', 'co_s', 'temp_s', 'humi_s', 'pressure_s']
col_xin_yang = ['aqi_x', 'pm2_5_x', 'pm10_x', 'so2_x', 'no2_x', 'co_x', 'temp_x', 'humi_x', 'pressure_x']
col_nan_yang = ['aqi_n', 'pm2_5_n', 'pm10_n', 'so2_n', 'no2_n', 'co_n', 'temp_n', 'humi_n', 'pressure_n']
col_zhou_kou = ['aqi_z', 'pm2_5_z', 'pm10_z', 'so2_z', 'no2_z', 'co_z', 'temp_z', 'humi_z', 'pressure_z']
col_shang_qiu = ['aqi_q', 'pm2_5_q', 'pm10_q', 'so2_q', 'no2_q', 'co_q', 'temp_q', 'humi_q', 'pressure_q']

san_men.columns = col_san_men
xin_yang.columns = col_xin_yang
nan_yang.columns = col_nan_yang
zhou_kou.columns = col_zhou_kou
shang_qiu.columns = col_shang_qiu

dataset = san_men.join(xin_yang)
dataset = dataset.join(nan_yang)
dataset = dataset.join(zhou_kou)
dataset = dataset.join(shang_qiu)
print(dataset.shape)

import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Add, TimeDistributed, MaxPooling2D, concatenate, \
    MaxPooling1D, Conv1D, LSTM, Flatten, RepeatVector, Reshape
from keras.layers import ConvLSTM2D, Conv2D, Dropout, UpSampling2D

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

var_origin = dataset.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(var_origin)
var = scaled


# 划分数据集，验证集，测试集
def splitData(var, per_val, per_test):
    num_val = int(len(var) * per_val)
    num_test = int(len(var) * per_test)
    train_size = int(len(var) - num_val - num_test)
    train_data = var[0:train_size]
    val_data = var[train_size:train_size + num_val]
    test_data = var[train_size + num_val:train_size + num_val + num_test]
    return train_data, val_data, test_data


train_data, val_data, test_data = splitData(var, 0.1, 0.1)
print('The length of train data, validation data and test data are:', len(train_data), ',', len(val_data), ',',
      len(test_data))

train_window = 240


def create_train_sequence(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_inout_sequence = create_train_sequence(train_data, train_window)
print("the length of train_sequence is: ", len(train_inout_sequence))


def create_val_sequence(train_data, val_data, tw):
    temp = np.concatenate((train_data, val_data))  # 先将训练集和测试集合并
    inout_seq = []
    L = len(val_data)
    for i in range(L):
        val_seq = temp[-(train_window + L) + i:-L + i]
        val_label = test_data[i:i + 1]
        inout_seq.append((val_seq, val_label))

    return inout_seq


# 注意，与上面创建train_data的sequence不同，
# 验证集数据只是label。其数据部分还是需要借助于train集中的数据，大小为一个窗口。而这一个窗口的数据并不会在训练过程中被使用
# 此时的label的shape是[1,40]。注意，真正的label只有这40个值中的前五个
val_inout_seq = create_val_sequence(train_data, val_data, train_window)
print('The total number of validation windows is', len(val_inout_seq))


def create_test_sequence(train_data, val_data, test_data, tw):
    temp = np.concatenate((train_data, val_data))  # 先将训练集和测试集合并
    temp = np.concatenate((temp, test_data))
    inout_seq = []
    L = len(test_data)
    for i in range(L):
        test_seq = temp[-(train_window + L) + i:-L + i]
        test_label = test_data[i:i + 1]
        inout_seq.append((test_seq, test_label))

    return inout_seq


test_inout_seq = create_test_sequence(train_data, val_data, test_data, train_window)
print('The total number of validation windows is', len(val_inout_seq))


# n_length = 40
# n_steps = 24
# 构建网络模型ConvLSTM
def model_build(train_x, train_y, n_steps, in_outputs):
    # [samples，timesteps，rows，cols，channels] = [train_x.shape[0] = 13776, 24, 10, 40,1]
    # n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    n_timesteps, n_features, n_outputs = train_x.shape[1], 1, train_y.shape[1]
    # train_x = train_x.reshape((train_x.shape[0], n_steps, int(n_timesteps / n_steps), n_length, n_features))
    # train_y = train_y.reshape((train_y.shape[0], train_y.shape[2], 1))
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu',
                         input_shape=(n_steps, 10, in_outputs, n_features)))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(in_outputs, activation='relu'))
    print(model.output_shape)

    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
    print(model.summary())
    return model


seqList = []
labelList = []
for seq, label in train_inout_sequence:
    seqList.append(seq)
    labelList.append(label)

seqList = np.array(seqList)
labelList = np.array(labelList)
n_steps = 24
n_outputs = 45
model = model_build(seqList, labelList, n_steps, n_outputs)
epochs_num = 15
batch_size_set = 1
weight_path = '../try_code/He_Nan_ConvLSTM_weight_2.h5'
# weight_path = ''
isTrain = False
if isTrain:
    with tf.device("/gpu:0"):
        train_x1, train_y1 = seqList, labelList
        train_x1 = train_x1.reshape((train_x1.shape[0], n_steps, 10, n_outputs, 1))
        train_y1 = labelList.reshape(labelList.shape[0], labelList.shape[2])
        # train_y1 = train_y1[:, :5]
        model.fit(train_x1, train_y1,
                  epochs=epochs_num, batch_size=batch_size_set, verbose=2)
    try:
        os.remove(weight_path)
    except:
        print(f'no such files in path')
    model.save_weights(weight_path)
else:
    model.summary()
    model.load_weights(weight_path)

test_x = []
test_y = []
for seq, label in test_inout_seq:
    test_x.append(seq)
    test_y.append(label)
test_x = np.array(test_x)
test_x = test_x.reshape(test_x.shape[0], 24, 10, n_outputs, 1)
test_y = np.array(test_y)
test_y = test_y.reshape(test_y.shape[0], test_y.shape[2])
test_y = test_y[:, :9]


def show_graph(yhat, test_y):
    # air_pollute_list = ['nox', 'no2', 'no', 'o3', 'pm2.5']
    air_pollute_list = ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'temp', 'humi', 'pressure']

    pyplot.figure(figsize=(28, 14))
    i = 1
    for column in air_pollute_list:
        pyplot.subplot(len(air_pollute_list), 1, i)
        pyplot.title(column, y=0.5, loc='right')
        pyplot.plot(yhat[:, i - 1], color='red', label='prediction')
        pyplot.plot(test_y[:, i - 1], color='blue', label='actual')
        i += 1
    pyplot.show()


yhat = model.predict(test_x)
yhat = yhat[:, :, :9]
yhat = yhat.reshape(yhat.shape[0], yhat.shape[2])


def acc15(yhat, test_y, idx):
    cnt = 0
    for i in range(int(len(test_y))):
        if test_y[i][idx] * 0.80 < yhat[i][idx] < test_y[i][idx] * 1.2:
            cnt += 1
    return cnt / int(len(test_y))


res = []
for i in range(0, test_y.shape[1], 1):
    acc = acc15(yhat, test_y, i)
    res.append(acc)

print(res)
print(yhat)
print(test_y)
