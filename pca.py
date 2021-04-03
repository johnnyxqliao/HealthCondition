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

# 计算健康指数
# dataframe = pd.DataFrame({'healthLevel': [feature]})
# dataframe.to_csv("./data/healthLevel.csv", index=False, sep=',')

# healthIndex = np.transpose(pca.fit_transform(X))[0]
# X = np.arange(1, np.size(healthIndex), 1)

# 健康衰退曲线数据
# data = np.transpose(pd.read_csv('./data/curve.csv').values)

# 计算健康等级
# healthLevel = levelDivide(feature)
# dataframe = pd.DataFrame({'healthLevel': [healthLevel]})
# dataframe.to_csv("./data/healthIndex.csv", index=False, sep=',')

# 计算最小和最大时间（到下一个健康等级）
# remainTime = nextLevelTime(healthLevel, feature, data)
# dataframe = pd.DataFrame({'maxTime & minTime': remainTime})
# dataframe.to_csv("./data/remainTime.csv", index=False, sep=',')

# 画图表示曲线
# plt.plot(data[0], data[1])
# plt.plot(X, [feature] * np.size(X))
# plt.show()


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
    return feature


basePath = 'data/predict/Data5-Cvt/'
# inputFile = [
#     '20190525044745-01-25600.csv',
#     '20190526061221-01-25600.csv',
#     '20190528201726-01-25600.csv',
#     '20190630155458-01-25600.csv',
#     '20190708102254-01-25600.csv',
#     '20190726093145-01-25600.csv',
#     '20190727004306-01-25600.csv',
#     '20190729130618-01-25600.csv',
#     '20190730205425-01-25600.csv',
#     '20190803044742-01-25600.csv'
# ]
inputFile = [
    '20190525044745-02-25600.csv',
    '20190526061221-02-25600.csv',
    '20190528201726-02-25600.csv',
    '20190630155458-02-25600.csv',
    '20190708102254-02-25600.csv',
    '20190726093145-02-25600.csv',
    '20190727004306-02-25600.csv',
    '20190729130618-02-25600.csv',
    '20190730205425-02-25600.csv',
    '20190803044742-02-25600.csv'
]
# inputFile = [
#     '20190525044745-03-25600.csv',
#     '20190526061221-03-25600.csv',
#     '20190528201726-03-25600.csv',
#     '20190630155458-03-25600.csv',
#     '20190708102254-03-25600.csv',
#     '20190726093145-03-25600.csv',
#     '20190727004306-03-25600.csv',
#     '20190729130618-03-25600.csv',
#     '20190730205425-03-25600.csv',
#     '20190803044742-03-25600.csv'
# ]
# inputFile = [
#     '20190525044745-04-25600.csv',
#     '20190526061221-04-25600.csv',
#     '20190528201726-04-25600.csv',
#     '20190630155458-04-25600.csv',
#     '20190708102254-04-25600.csv',
#     '20190726093145-04-25600.csv',
#     '20190727004306-04-25600.csv',
#     '20190729130618-04-25600.csv',
#     '20190730205425-04-25600.csv',
#     '20190803044742-04-25600.csv'
# ]
# inputFile = [
#     '20190525044745-05-25600.csv',
#     '20190526061221-05-25600.csv',
#     '20190528201726-05-25600.csv',
#     '20190630155458-05-25600.csv',
#     '20190708102254-05-25600.csv',
#     '20190726093145-05-25600.csv',
#     '20190727004306-05-25600.csv',
#     '20190729130618-05-25600.csv',
#     '20190730205425-05-25600.csv',
#     '20190803044742-05-25600.csv'
# ]
# inputFile = [
#     '20190525044745-06-25600.csv',
#     '20190526061221-06-25600.csv',
#     '20190528201726-06-25600.csv',
#     '20190630155458-06-25600.csv',
#     '20190708102254-06-25600.csv',
#     '20190726093145-06-25600.csv',
#     '20190727004306-06-25600.csv',
#     '20190729130618-06-25600.csv',
#     '20190730205425-06-25600.csv',
#     '20190803044742-06-25600.csv'
# ]
# inputFile = [
#     '20190525044745-07-25600.csv',
#     '20190526061221-07-25600.csv',
#     '20190528201726-07-25600.csv',
#     '20190630155458-07-25600.csv',
#     '20190708102254-07-25600.csv',
#     '20190726093145-07-25600.csv',
#     '20190727004306-07-25600.csv',
#     '20190729130618-07-25600.csv',
#     '20190730205425-07-25600.csv',
#     '20190803044742-07-25600.csv'
# ]
# inputFile = [
#     '20190525044745-08-25600.csv',
#     '20190526061221-08-25600.csv',
#     '20190528201726-08-25600.csv',
#     '20190630155458-08-25600.csv',
#     '20190708102254-08-25600.csv',
#     '20190726093145-08-25600.csv',
#     '20190727004306-08-25600.csv',
#     '20190729130618-08-25600.csv',
#     '20190730205425-08-25600.csv',
#     '20190803044742-08-25600.csv'
# ]

# 输入端2

# inputFile = [
#     '20190516114411-01-25600.csv',
#     '20190525030807-01-25600.csv',
#     '20190605094049-01-25600.csv',
#     '20190612182414-01-25600.csv',
#     '20190711180530-01-25600.csv',
#     '20190717075142-01-25600.csv',
#     '20190720195520-01-25600.csv',
#     '20190722083029-01-25600.csv',
#     '20190731235854-01-25600.csv',
#     '20190804215800-01-25600.csv'
# ]
# inputFile = [
#     '20190516114411-02-25600.csv',
#     '20190525030807-02-25600.csv',
#     '20190605094049-02-25600.csv',
#     '20190612182414-02-25600.csv',
#     '20190711180530-02-25600.csv',
#     '20190717075142-02-25600.csv',
#     '20190720195520-02-25600.csv',
#     '20190722083029-02-25600.csv',
#     '20190731235854-02-25600.csv',
#     '20190804215800-02-25600.csv'
# ]
# inputFile = [
#     '20190516114411-03-25600.csv',
#     '20190525030807-03-25600.csv',
#     '20190605094049-03-25600.csv',
#     '20190612182414-03-25600.csv',
#     '20190711180530-03-25600.csv',
#     '20190717075142-03-25600.csv',
#     '20190720195520-03-25600.csv',
#     '20190722083029-03-25600.csv',
#     '20190731235854-03-25600.csv',
#     '20190804215800-03-25600.csv'
# ]
# inputFile = [
#     '20190516114411-04-25600.csv',
#     '20190525030807-04-25600.csv',
#     '20190605094049-04-25600.csv',
#     '20190612182414-04-25600.csv',
#     '20190711180530-04-25600.csv',
#     '20190717075142-04-25600.csv',
#     '20190720195520-04-25600.csv',
#     '20190722083029-04-25600.csv',
#     '20190731235854-04-25600.csv',
#     '20190804215800-04-25600.csv'
# ]
# inputFile = [
#     '20190516114411-05-25600.csv',
#     '20190525030807-05-25600.csv',
#     '20190605094049-05-25600.csv',
#     '20190612182414-05-25600.csv',
#     '20190711180530-05-25600.csv',
#     '20190717075142-05-25600.csv',
#     '20190720195520-05-25600.csv',
#     '20190722083029-05-25600.csv',
#     '20190731235854-05-25600.csv',
#     '20190804215800-05-25600.csv'
# ]
outputResult = []
# 修正系数（缩尺数据归一化到实验室数据）
zoom_factor = 0.71
translation_factor = 0.86
for index, file in enumerate(inputFile):
    print("开始执行第", index)
    outputResult.append(getHealthIndex(basePath + file, zoom_factor, translation_factor))
print(outputResult)
plt.plot(np.linspace(1, len(outputResult), len(outputResult)), outputResult)
plt.show()
