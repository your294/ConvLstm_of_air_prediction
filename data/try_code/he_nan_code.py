import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot
import os

from matplotlib.ticker import MultipleLocator

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

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Add, TimeDistributed, MaxPooling2D, concatenate, \
    MaxPooling1D, Conv1D, LSTM, Flatten, RepeatVector, Reshape, Softmax, Lambda, Attention
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


# reverse_transform 逆归一化
def reverse_transform(data, sca_data):
    mx, mn = np.max(data), np.min(data)
    std_data = np.full(len(sca_data), 0)
    for idx in range(len(sca_data)):
        std_data[idx] = sca_data[idx] * (mx - mn) + mn
    return std_data


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
pred_time = 12


def create_train_sequence(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - pred_time):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + pred_time, :9]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_inout_sequence = create_train_sequence(train_data, train_window)
print("the length of train_sequence is: ", len(train_inout_sequence))


def create_val_sequence(train_data, val_data, tw):
    temp = np.concatenate((train_data, val_data))  # 先将训练集和测试集合并
    inout_seq = []
    L = len(val_data)
    for i in range(0, L - pred_time, pred_time):
        val_seq = temp[-(tw + L) + i + pred_time:-L + i + pred_time]
        val_label = test_data[i:i + pred_time, :9]
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
    for i in range(0, L - pred_time, pred_time):
        test_seq = temp[-(tw + L) + i + pred_time:-L + i + pred_time]
        test_label = test_data[i:i + pred_time, :9]
        inout_seq.append((test_seq, test_label))
    return inout_seq


test_inout_seq = create_test_sequence(train_data, val_data, test_data, train_window)
print('The total number of test windows is', len(test_inout_seq))


def new_model_build(train_x, train_y, n_steps, in_outputs):
    train_x = train_x.reshape((train_x.shape[0], n_steps, 10, in_outputs, 1))
    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1])
    n_timesteps, n_features, n_outputs = train_x.shape[1], 1, 1
    inputs = Input(shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3], train_x.shape[4]))
    conv_lstm1 = ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', return_sequences=True)(inputs)
    conv_lstm2 = ConvLSTM2D(filters=32, kernel_size=(1, 3), activation='relu')(conv_lstm1)
    # attention_pre = Dense(64, name='attention_vec')(conv_lstm1)  # [b_size,maxlen,1]
    # attention_probs = Softmax()(attention_pre)  # [b_size,maxlen,1]
    # attention_mul = Lambda(lambda x: x[0] * x[1])([attention_probs, conv_lstm1])
    flatten = Flatten()(conv_lstm2)
    repeater = RepeatVector(1)(flatten)
    lstm = LSTM(200, activation='relu', return_sequences=True)(repeater)
    dense1 = TimeDistributed(Dense(100, activation='relu'))(lstm)
    x_output = TimeDistributed(Dense(pred_time, activation='relu'))(dense1)
    model = Model(inputs=inputs, outputs=x_output)

    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
    model.summary()
    plot_model(model, to_file='./img/conv_lstm2.png', show_shapes=True, show_layer_names='True', rankdir="TB")
    return model


idx_target = 5
seqList = []
labelList = []
for seq, label in train_inout_sequence:
    seqList.append(seq)
    labelList.append(label[:, idx_target])

seqList = np.array(seqList)
labelList = np.array(labelList)
n_steps = 24
n_outputs = 45

model = new_model_build(seqList, labelList, n_steps, n_outputs)
epochs_num = 3
batch_size_set = 1
weight_path = f'./weight_of_target/weight_{idx_target}.h5'
# weight_path = ''
isTrain = False
if isTrain:
    with tf.device("/gpu:0"):
        train_x1, train_y1 = seqList, labelList
        train_x1 = train_x1.reshape((train_x1.shape[0], n_steps, 10, n_outputs, 1))
        model.fit(train_x1, train_y1,
                  epochs=epochs_num, batch_size=batch_size_set, verbose=2)
    try:
        os.remove(weight_path)
    except:
        print(f'no such files in path')
    try:
        model.save_weights(weight_path)
    except:
        model.save(f'./weight_{idx_target}.h5')
else:
    model.summary()
    model.load_weights(weight_path)

test_x = []
test_y = []
for seq, label in test_inout_seq:
    test_x.append(seq)
    test_y.append(label[:, idx_target])
test_x = np.array(test_x)
test_x = test_x.reshape(test_x.shape[0], 24, 10, n_outputs, 1)
test_y = np.array(test_y)


# def show_hours_graph(yhat, test_y):
#     air_pollute_list = ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'temp', 'humi', 'pressure']
#     pyplot.figure(figsize=(14, 14))
#     for hours in range(0, 12, 1):
#         pyplot.subplot(pred_time, 1, hours + 1)
#         pyplot.title(f'{hours + 1}h', y=0.5, loc='right')
#         pyplot.plot(yhat[:, hours], color='red', label='prediction')
#         pyplot.plot(test_y[:, hours], color='blue', label='fact')
#     pyplot.show()
#     pyplot.savefig(f'./img/new_{idx_target}.png')


def show_graph(predict, test):
    air_pollute_list = ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'temp', 'humi', 'pressure']
    pyplot.figure(figsize=(14, 6))

    # set the val of axis
    ax = pyplot.gca()
    # 把x轴的刻度间隔设置为1，并存在变量里
    x_major_locator = MultipleLocator(120)
    # y_major_locator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    # pyplot.ylim(0, 0.9)
    pyplot.xlim(0, len(test_y) * 12)
    pyplot.xlabel("test_hours")
    pyplot.ylabel(f'{air_pollute_list[idx_target]}')

    pyplot.title(f'test_{air_pollute_list[idx_target]}_{len(test_y)}_h', y=0.5, loc='center')
    pyplot.plot(predict, color='red', label='predict')
    pyplot.plot(test, color='blue', label='fact')
    pyplot.savefig(f'./img/{air_pollute_list[idx_target]}_pic.png')
    pyplot.show()


yhat = model.predict(test_x)
yhat = yhat.reshape(yhat.shape[0], yhat.shape[2])
predict_hours_of_96 = []
test_hours_of_96 = []
for i in range(len(test_y)):
    for j in range(12):
        predict_hours_of_96.append(yhat[i][j])
        test_hours_of_96.append(test_y[i][j])
predict_res, test_res = np.array(predict_hours_of_96), np.array(test_hours_of_96)
pre, test = reverse_transform(var_origin[:, 0].T, predict_res), reverse_transform(var_origin[:, 0].T, test_res)
show_graph(pre, test)

cnt = 0
for i in range(len(test_hours_of_96)):
    if test_hours_of_96[i] * 0.80 <= predict_hours_of_96[i] <= test_hours_of_96[i] * 1.20:
        cnt += 1
print(cnt / len(test_hours_of_96))
# show_hours_graph(yhat, test_y)

# cal acc15
# true_cnt = [0 for _ in range(pred_time)]
# row, col = test_y.shape[0], test_y.shape[1]
# for i in range(col):
#     for j in range(row):
#         if test_y[j][i] * 0.80 <= yhat[j][i] <= test_y[j][i] * 1.20:
#             true_cnt[i] += 1
#     print(true_cnt[i] / row)
