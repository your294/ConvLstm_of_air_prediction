import pandas as pd
import numpy as np
from matplotlib import pyplot
import os

print(os.getcwd())

# 添加数据列扩充数据
he_nan_pos = ['三门峡', '信阳', '南阳', '周口', '商丘']
san_men_w = pd.read_excel('../he_nan_data/weather/三门峡_2.xlsx', index_col=0)
xin_yang_w = pd.read_excel('../he_nan_data/weather/信阳_2.xlsx', index_col=0)
nan_yang_w = pd.read_excel('../he_nan_data/weather/南阳_2.xlsx', index_col=0)
zhou_kou_w = pd.read_excel('../he_nan_data/weather/周口_2.xlsx', index_col=0)
shang_qiu_w = pd.read_excel('../he_nan_data/weather/商丘_2.xlsx', index_col=0)

san_men_p = pd.read_excel('../he_nan_data/pollute/三门峡_2.xlsx', index_col=0)
xin_yang_p = pd.read_excel('../he_nan_data/pollute/信阳_2.xlsx', index_col=0)
nan_yang_p = pd.read_excel('../he_nan_data/pollute/南阳_2.xlsx', index_col=0)
zhou_kou_p = pd.read_excel('../he_nan_data/pollute/周口_2.xlsx', index_col=0)
shang_qiu_p = pd.read_excel('../he_nan_data/pollute/商丘_2.xlsx', index_col=0)

pdata_list = [san_men_p, xin_yang_p, nan_yang_p, zhou_kou_p, shang_qiu_p]
wdata_list = [san_men_w, xin_yang_w, nan_yang_w, zhou_kou_w, shang_qiu_w]
weather_list = ['temp', 'humi', 'pressure']
for i in range(0, int(len(pdata_list)), 1):
    pdata, wdata = pdata_list[i], wdata_list[i]
    for w in weather_list:
        pdata.insert(pdata.shape[1], w, wdata[w])
    pdata.to_excel(f'../he_nan_data/pollute/{he_nan_pos[i]}.xlsx', index=False)
