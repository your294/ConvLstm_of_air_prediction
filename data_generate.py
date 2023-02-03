import datetime

import pandas as pd

# 17个地点进行编码
position_arr = ["三门峡", "信阳", "南阳", "周口", "商丘",
                "安阳", "平顶山", "开封", "新乡", "洛阳",
                "漯河", "濮阳", "焦作", "许昌", "郑州",
                "驻马店", "鹤壁"]

data_weather = pd.read_excel("data/河南气象数据(2019-01~2021-09).xlsx")
data_pollute = pd.read_excel("data/河南空气质量数据(2019-01~2021-09).xlsx")
two_years = 2 * 365 * 24




def data_generate(df):
    # 18地点，2年时间
    length = two_years * 18
    dataFrameList = []

    for i in range(0, 18):
        new_df = pd.DataFrame(columns=df.columns)
        for j in range(0, length):
            if df.loc[j, 'cityname'] == position_arr[i]:
                element = df.loc[j, :].to_frame()
                element = pd.DataFrame(element.values.T, columns=element.index)
                new_df = pd.concat([new_df, element])
        dataFrameList.append(new_df)

    return dataFrameList


dfList = data_generate(data_pollute)
dfList2 = data_generate(data_weather)
for i in range(0, 18):
    dfList[i].to_excel(f'data/generate/pollute_{position_arr[i]}.xlsx', index=False)
    dfList2[i].to_excel(f'data/generate/weather_{position_arr[i]}.xlsx', index=False)
