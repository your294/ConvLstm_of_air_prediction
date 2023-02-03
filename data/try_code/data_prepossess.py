import pandas as pd
import numpy as np
from matplotlib import pyplot

pre_path = '../London_data/'
Bloomsbury = pd.read_csv(pre_path + 'Bloomsbury.csv')
Eltham = pd.read_csv(pre_path + 'Eltham.csv')
Harlington = pd.read_csv(pre_path + 'Harlington.csv')
Marylebone_Road = pd.read_csv(pre_path + 'Marylebone_Road.csv')
N_Kensington = pd.read_csv(pre_path + 'N_Kensington.csv')

site_names = ['Bloomsbury', 'Eltham', 'Harlington',
              'Marylebone_Road', 'N_Kensington']
air_pollutants_list = ['nox', 'no2', 'no', 'o3', 'pm2.5', 'ws', 'wd', 'air_temp']
sites_dict = {
    'Bloomsbury': Bloomsbury,
    'Eltham': Eltham,
    'Harlington': Harlington,
    'Marylebone_Road': Marylebone_Road,
    'N_Kensington': N_Kensington
}

for name in site_names:
    print(name + ':')
    print('O3为空的数量:', sites_dict[name]['o3'].isnull().value_counts().values[1])
    print('nox为空的数量:', sites_dict[name]['nox'].isnull().value_counts().values[1])
    print('no2为空的数量:', sites_dict[name]['no2'].isnull().value_counts().values[1])
    print('pm2.5为空的数量:', sites_dict[name]['pm2.5'].isnull().value_counts().values[1])
    print('ws为空的数量:', sites_dict[name]['ws'].isnull().value_counts().values[1])
    print('wd为空的数量:', sites_dict[name]['wd'].isnull().value_counts().values[1])
    print('air_temp为空的数量:', sites_dict[name]['air_temp'].isnull().value_counts().values[1])


def show_graph(site):
    dataset = sites_dict[site]
    value = dataset.values
    columns = [4, 5, 6, 7, 8, 9, 10, 13]

    pyplot.figure(figsize=(14, 14))
    i = 1
    for column in columns:
        pyplot.subplot(len(columns), 1, i)
        pyplot.plot(value[:, column])
        pyplot.title(dataset.columns[column], y=0.5, loc='right')
        i += 1
    pyplot.show()


# 经纬度
coordinate_dic = {'Bloomsbury': [51.52229, -0.125889],
                  'Eltham': [51.45258, 0.070766],
                  'Harlington': [51.48879, -0.441614],
                  'Marylebone_Road': [51.52253, -0.154611],
                  'N_Kensington': [51.52105, -0.213492]
                  }

# copy data for some reason
import copy

Bloomsbury_copy = copy.deepcopy(Bloomsbury)
Eltham_copy = copy.deepcopy(Eltham)
Harlington_copy = copy.deepcopy(Harlington)
Marylebone_Road_copy = copy.deepcopy(Marylebone_Road)
N_Kensington_copy = copy.deepcopy(N_Kensington)

copy_dic = {'Bloomsbury': Bloomsbury_copy,
            'Eltham': Eltham_copy,
            'Harlington': Harlington_copy,
            'Marylebone_Road': Marylebone_Road_copy,
            'N_Kensington': N_Kensington_copy
            }

import math


def interpolation(lon, lat, lst, P=2):
    """
    :param lon:要插值的点的x
    :param lat:要插值的点的y
    :param lst:lst是已有数据的数组，结构为：[[x1，y1，z1]，[x2，y2，z2]，...]
    :return:返回值是插值点的值
    """
    p0 = [lon, lat]
    sum0 = 0
    sum1 = 0
    temp = []
    # 遍历获取该点距离所有采样点的距离
    for point in lst:
        if lon == point[0] and lat == point[1]:
            return point[2]
        Di = distance(p0, point)
        # new出来一个对象，不然会改变原来lst的值
        ptn = copy.deepcopy(point)
        ptn.append(Di)
        temp.append(ptn)

    # 根据上面ptn.append（）的值由小到大排序
    temp1 = sorted(temp, key=lambda point: point[3])
    # 遍历排序的前15个点，根据公式求出sum0 and sum1
    for point in temp1[0:]:
        sum0 += point[2] / math.pow(point[3], P)
        sum1 += 1 / math.pow(point[3], P)

    return sum0 / sum1


# 计算两点间的距离
def distance(p, pi):
    dis = (p[0] - pi[0]) * (p[0] - pi[0]) + (p[1] - pi[1]) * (p[1] - pi[1])
    m_result = math.sqrt(dis)
    return m_result


def full_fill_missing(site):
    ls_temp = copy.deepcopy(site_names)
    ls_temp.remove(site)

    for i in sites_dict[site].index.tolist():
        for j in air_pollutants_list:
            # iloc 根据label定位数据
            if np.isnan(sites_dict[site].iloc[i][j]):
                ls_temp2 = []
                for m in ls_temp:
                    if (not np.isnan(sites_dict[m].iloc[i][j])):
                        temp_value = copy.deepcopy(coordinate_dic[m])
                        temp_value.append(sites_dict[m].iloc[i][j])
                        ls_temp2.append(temp_value)
                if len(ls_temp2) != 0:  # 防止所有监测站都为空
                    copy_dic[site].loc[i, j] = interpolation(coordinate_dic[site][0], coordinate_dic[site][1], ls_temp2)


full_fill_missing('Bloomsbury')
full_fill_missing('Eltham')
full_fill_missing('Harlington')
full_fill_missing('Marylebone_Road')
full_fill_missing('N_Kensington')

# 线性插值
for i in site_names:
    copy_dic[i] = copy_dic[i].interpolate(method='linear', axis=0)

empty = [[], [], [], [], [], [], [], []]  # 用来存放空气污染物为nan值的索引
for i in range(len(air_pollutants_list)):
    em = Bloomsbury[air_pollutants_list[i]]
    ls = em[np.isnan(em)].index.tolist()
    empty[i] = ls


def show_graph2(site):
    dataset = sites_dict[site]
    dataset2 = copy_dic[site]

    air_pollutants_list = ['nox', 'no2', 'no', 'o3', 'pm2.5', 'ws', 'wd', 'air_temp']

    # plot each column
    pyplot.figure(figsize=(14, 14))
    i = 1
    for column in air_pollutants_list:
        pyplot.subplot(len(air_pollutants_list), 1, i)
        pyplot.plot(dataset.loc[:, column].values, linewidth=0.5)
        pyplot.title(column, y=0.5, loc='right')
        pyplot.scatter(empty[i - 1], dataset2.loc[empty[i - 1], column].values, color='red', s=0.4)
        i += 1

    pyplot.show()


show_graph2('Bloomsbury')
pyplot.close()

# 消除噪声点
from scipy.signal import savgol_filter

for i in site_names:
    for j in air_pollutants_list:
        data = copy_dic[i].loc[:, j].values
        copy_dic[i].loc[:, j] = savgol_filter(data, 121, 3, mode='nearest')


def show_graph3(site):
    dataset = sites_dict[site]
    dataset2 = copy_dic[site]

    # plot each column
    pyplot.figure(figsize=(14, 14))
    i = 1
    for column in air_pollutants_list:
        pyplot.subplot(len(air_pollutants_list), 1, i)
        pyplot.plot(dataset2.loc[:, column].values, linewidth=0.5, color='b')
        pyplot.title(column, y=0.5, loc='right')
        #     pyplot.scatter(empty[i-1],dataset2.loc[empty[i-1], column].values,color='red',s=0.4)
        i += 1
    pyplot.show()


show_graph3('Bloomsbury')

for site in site_names:
    copy_dic[site] = copy_dic[site][['site', 'code', 'date', 'nox', 'no2', 'no', 'o3', 'pm2.5', 'ws', 'wd', 'air_temp']]

copy_dic['Bloomsbury'].to_csv(pre_path + "clean/Bloomsbury_clean.csv", index=False)
copy_dic['Eltham'].to_csv(pre_path + "clean/Eltham_clean.csv", index=False)
copy_dic['Harlington'].to_csv(pre_path + "clean/Harlington_clean.csv", index=False)
copy_dic['Marylebone_Road'].to_csv(pre_path + "clean/Marylebone_Road_clean.csv", index=False)
copy_dic['N_Kensington'].to_csv(pre_path + "clean/N_Kensington_clean.csv", index=False)
