import pandas as pd
import numpy as np
from matplotlib import pyplot
import datetime
import os

import tensorflow as tf

print(tf.__version__) # 输出版本

he_nan_pos = ['三门峡', '信阳', '南阳', '周口', '商丘']
start_time = datetime.datetime(2019, 1, 1, 0, 0)
delta = datetime.timedelta(hours=1)
times = 365 * 24

base_path = '../he_nan_data/'
data_pollute = [] * 5
for file in os.listdir(base_path + 'pollute'):
    data_pollute.append(pd.read_excel(base_path + 'pollute/' + file, index_col=0))
data_weather = [] * 5
for file in os.listdir(base_path + 'weather'):
    data_weather.append(pd.read_excel(base_path + 'weather/' + file, index_col=0))


for i in range(int(len(he_nan_pos))):
    new_df = pd.DataFrame(columns=data_weather[0].columns)
    j = 0
    p = 0
    while j < times and p < int(len(data_weather[0])):
        cur_time = start_time + j * delta
        if data_weather[i].loc[p, 'time'] == cur_time:
            new_df = new_df.append(data_weather[i].loc[p].to_frame().T, ignore_index=True)
            j += 1
            p += 1
        else:
            base = data_weather[i].loc[p, 'time']
            dtime = int((base - cur_time) / delta)
            for k in range(0, dtime, 1):
                new_df = pd.concat([new_df, pd.DataFrame([[np.NaN]*9],columns = new_df.columns)], ignore_index=True)
                new_df.loc[int(len(new_df)) - 1, 'time'] = cur_time + k * delta
                new_df.loc[int(len(new_df)) - 1, 'cityname'] = he_nan_pos[i]
                j += 1
    new_df.to_excel(base_path + f'weather/{he_nan_pos[i]}.xlsx')


