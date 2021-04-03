import os

import numpy as np
# from normalization import Normalization
import csv
import pandas as pd
import scipy as sc

from sklearn.decomposition import PCA

# X = np.array([[10, 6], [11, 4], [8, 5], [3, 3], [2, 2.8], [1, 1]])
# pca = PCA(n_components=2, svd_solver='full')
# pca.fit(X)
# print("主成分占比：", pca.explained_variance_ratio_)
# print("降维后的矩阵：", pca.fit_transform(X))

import matplotlib.pyplot as plt

# 读取多个csv数据
# data = np.transpose(pd.read_csv('./data/curve.csv').values)

#
# dataframe = pd.DataFrame({'healthLevel': [feature]})
# dataframe.to_csv("./data/healthLevel.csv", index=False, sep=',')
# 输出预测结果到文件
# def toCSV(result, path):
#     with open(path, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(result)


#
# mu, sigma = 0, 500
#
# x = np.arange(1, 100, 0.1)  # x axis
# z = np.random.normal(mu, sigma, len(x))  # noise
# y = x ** 2 + z  # data
# plt.plot(x, y)  # it include some noise
# plt.show()
# from scipy.signal import lfilter
#
# n = 2  # the larger n is, the smoother curve will be
# b = [1.0 / n] * n
# print(b)
# a = 1
# yy = lfilter(b, a, y)
# plt.plot(x, yy)  # smooth by filter
#
# plt.show()
# path = "./data/Bearing2_1"

# def readAllFiles(filePath):
#     files = os.listdir(filePath)
#     # 文件按照文件名进行排序
#     files.sort(key=lambda x: int(x[:-4]))
#     verticalData = []
#     # 遍历文件夹
#     for file in files:
#         position = path + '/' + file
#         test = np.loadtxt(position, dtype=np.str)
#         test = np.delete(test, 0, 0)
#         for index in test:
#             # 将水平传感器状态数据转化为浮点型
#             verticalData.append(float(index.split(",")[0]))
#     return verticalData


# print(readAllFiles(path))
# self = np.array([1, 2, 3])
# maxFeature, minFeature = np.max(self), np.min(self)
# 矩阵归一化
# normalMatrix = (self - minFeature) / (maxFeature - minFeature)
# Z = (self - minFeature) / (maxFeature - minFeature)
# print(Z)
#
# test = np.array([0.0773, 0.0091, 0.2364, 0.0636, 0.0773, 0.0500, 0.2455, 0.0545, 0.0136, 0.0409, 0.0364, 0.0182, 0.0091, 0.0318])
# print(np.sort(test))
# self = np.array([[1, 2, 3]])
# temp =
# print(np.c_(self,temp))
# print(np.ones(3))

# a = [2, 3, 4, 5, 6, 7, 5, 43, 1]
# a_another = [2, 2, 2, 2, 2, 2, 2, 2, 2]
# b =
# b = [1, 1, 1]
# print(np.r_[a, np.array([b])])
# print(a+1)
# print(np.delete(a,0))
# X =np.arange(1, 10,0)
# print(len(a) / 3)
# print(a[2:4:1])
# print(a[1, round(len(a) / 3), 1])
# plt.plot(X,a)
# plt.plot(X,a_another)
#
# plt.show().
# toCSV([a,a_another], './curve.csv')
# dataframe = pd.DataFrame({'a_name': a, 'b_name': a_another})
# dataframe.to_csv("./curve.csv", index=False, sep=',')
#
# data = pd.read_csv('./curve.csv')
# data = np.transpose(pd.read_csv('data/curve.csv').values)
# print(data)
# aa = np.loadtxt("./curve.csv", dtype=np.str)
# aa = np.loadtxt(aa, dtype=np.float128)
# print(aa)
# import numpy as np
# import shapely.geometry as SG

# x = np.array([712,653,625,605,617,635,677,762,800,872,947,1025,1111,1218,1309, 500])
# y = np.array([2022,1876,1710,1544,1347,1309,1025,995,850,723,705,710,761,873,1050, 2000])
# data = np.transpose(pd.read_csv('./data/curve.csv').values)
#
# line = SG.LineString(list(zip(data[0], data[1])))
# # yline = SG.LineString([(800, 0), (800, 2000)])
# xline = SG.LineString([(0, 0.5), (4000, 0.5)])
# coords = np.array(line.intersection(xline))
# print(coords)
# plt.plot(data[0], data[1])
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# data = np.transpose(pd.read_csv('./data/curve.csv').values)
#
# # plt.plot(data[0], data[1])
# x1 = np.array(data[0])
# y1 = np.array(data[1])
#
# x_start = np.min(x1)
# x_end = np.max(x1) + 1
#
# x_line = x1.copy()
# y_line = [0.8] * np.size(x_line)
#
# y = y1 - y_line
# nLen = len(x1)
# xzero = np.zeros((nLen,))
# yzero = np.zeros((nLen,))
# for i in range(nLen - 1):
#     if np.dot(y[i], y[i + 1]) == 0:  # %等于0的情况
#         if y[i] == 0:
#             xzero[i] = i
#             yzero[i] = 0
#         if y[i + 1] == 0:
#             xzero[i + 1] = i + 1
#             yzero[i + 1] = 0
#     elif np.dot(y[i], y[i + 1]) < 0:  # %一定有交点，用一次插值
#         yzero[i] = np.dot(abs(y[i]) * y_line[i + 1] + abs(y[i + 1]) * y_line[i], 1 / (abs(y[i + 1]) + abs(y[i])))
#         xzero[i] = yzero[i]
#     else:
#         pass
#
# for i in range(nLen):
#     if xzero[i] == 0 and (yzero[i] == 0):  # %除掉不是交点的部分
#         xzero[i] = np.nan
#         yzero[i] = np.nan
#
# print(xzero)
# print(yzero)
#
# plt.plot(x1, y1)
# plt.plot(x_line, y_line, xzero, yzero)
# plt.show()
# a = [2,4,7,8]
# b = [4,8,12,16]
# c= [-4,-8,-12,-16]
# a = np.array(a)
# b = np.array(b)
# c = np.array(c)
# print(sc.stats.pearsonr(a, b))  # (1.0,0.0)第一个数代表Pearson相关系数
# print(sc.stats.pearsonr(a, c))
#   # (-1.0,0.0)
# for i in 5:
#     print("str(5)", i, 5)
# baseFilePath = "data/predict/Data5-Cvt"
# num = 2
# index = 1
#
# pathDir = os.listdir(baseFilePath)
#
# for allDir in pathDir:
#     child = os.path.join('%s%s' % (baseFilePath, allDir))
#     print("路径下所有文件", child)
print(np.linspace(1, len([1, 1, 1, 3, 4]), len([1, 1, 1, 3, 4])))
