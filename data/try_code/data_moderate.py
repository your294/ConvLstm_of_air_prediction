import pandas as pd
import numpy as np
from matplotlib import pyplot
import datetime
import os

weather_data = pd.read_excel("../河南气象数据(2019-01~2021-09).xlsx")
pollution_data = pd.read_excel("../河南空气质量数据(2019-01~2021-09).xlsx")
# test_data = pd.read_excel("../test_pollution.xlsx")

he_nan_pos = ['三门峡', '信阳', '南阳', '周口', '商丘']
start_time = datetime.datetime(2019, 1, 1, 0, 0)
delta = datetime.timedelta(hours=1)
times = 365 * 24


def generate_files(data_frame, string):
    df_arr = [] * 5
    for pos in he_nan_pos:
        new_df = pd.DataFrame()
        for i in range(int(len(data_frame))):
            if data_frame.loc[i, 'cityname'] == pos:
                new_df = new_df.append(data_frame.loc[i].to_frame().T, ignore_index=True)
        df_arr.append(new_df)
    idx = 0
    for df in df_arr:
        df.to_excel(f'../he_nan_data/{string}/{he_nan_pos[idx]}_{string}.xlsx')
        idx += 1


generate_files(pollution_data, "pollute")
generate_files(weather_data, "weather")

