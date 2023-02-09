import pandas as pd
import numpy as np
from matplotlib import pyplot
import datetime
import os


he_nan_pos = ['三门峡', '信阳', '南阳', '周口', '商丘']
start_time = datetime.datetime(2019, 1, 1, 0, 0)
delta = datetime.timedelta(hours=1)
times = 365 * 24 * 2

base_path = '../he_nan_data/'
data_pollute = [] * 5
index = 0
for file in os.listdir(base_path + 'pollute'):
    if index >= 5:
        break
    if file == f'{he_nan_pos[index]}_pollute.xlsx':
        data_pollute.append(pd.read_excel(base_path + 'pollute/' + file))
        index += 1
data_weather = [] * 5
index = 0
for file in os.listdir(base_path + 'weather'):
    if index >= 5:
        break
    if file == f'{he_nan_pos[index]}_weather.xlsx':
        data_weather.append(pd.read_excel(base_path + 'weather/' + file))
        index += 1


def generate_data(data_list, string):
    for i in range(int(len(he_nan_pos))):
        new_df = pd.DataFrame(columns=data_list[0].columns)
        j = 0
        p = 0
        while j < times and p < int(len(data_list[0])):
            cur_time = start_time + j * delta
            if data_list[i].loc[p, 'time'] == cur_time:
                new_df = new_df.append(data_list[i].loc[p].to_frame().T, ignore_index=True)
                j += 1
                p += 1
            else:
                base = data_list[i].loc[p, 'time']
                dtime = int((base - cur_time) / delta)
                for k in range(0, dtime, 1):
                    new_df = pd.concat([new_df, pd.DataFrame([[np.NaN] * 9], columns=new_df.columns)],
                                       ignore_index=True)
                    new_df.loc[int(len(new_df)) - 1, 'time'] = cur_time + k * delta
                    new_df.loc[int(len(new_df)) - 1, 'cityname'] = he_nan_pos[i]
                    j += 1
        new_df.to_excel(base_path + f'{string}/{he_nan_pos[i]}_2.xlsx')


generate_data(data_pollute, "pollute")
generate_data(data_weather, "weather")
