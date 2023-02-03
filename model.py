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
    MaxPooling1D, Conv1D, LSTM, Flatten, SeparableConv2D
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

# 17个地点进行编码
position_arr = ["三门峡", "信阳", "南阳", "周口", "商丘",
                "安阳", "平顶山", "开封", "新乡", "洛阳",
                "漯河", "濮阳", "焦作", "许昌", "郑州",
                "驻马店", "鹤壁"]

# 只取5个地点
pre = "data/generate/"
pollute_files = os.listdir(pre + "pollute")
weather_files = os.listdir(pre + "weather")
idx = 0
pollute_excel = []
for name in pollute_files:
    if idx == 5:
        break
    df = pd.read_excel(pre + "pollute/" + name)
    pollute_excel.append(df)
    idx = idx + 1

weather_excel = [], idx = 0
for name in weather_files:
    if idx == 5:
        break
    df = pd.read_excel(pre + "weather/" + name)
    weather_excel.append(df)
    idx = idx + 1

print(pollute_excel)
print(weather_excel)


# 构建网络模型
def model_build(input_shape, output_shape, build=True):
    x_in = Input(shape=input_shape)
    x_output = Conv2D(output_shape[-1], (1, 1), padding='same', activation='linear')(x_in)
    model = Model(x_in, x_output)
    model.compile(loss='mse', optimizer='adam')
    if build:
        model.summary()
    return model
