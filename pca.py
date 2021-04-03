import numpy as np

from healthLevel import levelDivide, nextLevelTime
from normalization import Normalization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc

featureSelected = [0, 8, 9, 2, 5]


# X = np.array(Normalization.X_train)
# X = np.row_stack((X, Normalization.X_predict))
# for i in range(np.size(X[0])):
#     temp = X[:, i]
#     print("第个参数的相关系数为", i, sc.stats.pearsonr(temp, np.arange(np.size(temp))))
# X = X[:, featureSelected]
# print("特征矩阵：", X)
# pca = PCA(n_components=2, svd_solver='full')
# pca.fit(X)
# print("主成分占比：", pca.explained_variance_ratio_)
# # print("降维后的矩阵:：", pca.fit_transform(X))
# feature = pca.fit_transform(X)[-1][0]
# print("降维后的特征指数HI:：", feature)

# 补充函数（可以删除）
def getHealthIndex(filePath, zoomFactor, translationFactor):
    normalization = Normalization(filePath, zoomFactor, translationFactor)
    X = np.array(normalization.X_train)
    X = np.row_stack((X, normalization.X_predict))
    for i in range(np.size(X[0])):
        temp = X[:, i]
        print("第个参数的相关系数为", i, sc.stats.pearsonr(temp, np.arange(np.size(temp))))
    X = X[:, featureSelected]
    print("特征矩阵：", X)
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(X)
    print("主成分占比：", pca.explained_variance_ratio_)
    feature = pca.fit_transform(X)[-1][0]
    print("降维后的特征指数HI:：", feature)
    # 计算健康指数
    dataframe = pd.DataFrame({'healthLevel': [feature]})
    dataframe.to_csv("./data/healthLevel.csv", index=False, sep=',')

    healthIndex = np.transpose(pca.fit_transform(X))[0]
    X = np.arange(1, np.size(healthIndex), 1)

    # 健康衰退曲线数据
    data = np.transpose(pd.read_csv('./data/curve.csv').values)

    # 计算健康等级
    healthLevel = levelDivide(feature)
    dataframe = pd.DataFrame({'healthLevel': [healthLevel]})
    dataframe.to_csv("./data/healthIndex.csv", index=False, sep=',')

    # 计算最小和最大时间（到下一个健康等级）
    remainTime = nextLevelTime(healthLevel, feature, data)
    dataframe = pd.DataFrame({'maxTime & minTime': remainTime})
    dataframe.to_csv("./data/remainTime.csv", index=False, sep=',')

    # 画图表示曲线
    plt.plot(data[0], data[1])
    plt.plot(X, [feature] * np.size(X))
    plt.show()


basePath = 'data/predict/'
inputFile = ['bearData.txt']

outputResult = []
# 修正系数（缩尺数据归一化到实验室数据）
zoom_factor = 0.71
translation_factor = 0.86
for index, file in enumerate(inputFile):
    outputResult.append(getHealthIndex(basePath + file, zoom_factor, translation_factor))
