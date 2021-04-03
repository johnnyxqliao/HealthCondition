# coding: utf-8
# import os
import numpy as np
from sklearn import preprocessing
from preprocessing import getFeature
from preprocessing import cut
import csv
import pandas as pd

# 读取文件，并转化为数组（长度15）
# predictPath = 'data/predict/Data5-Cvt/20190525044745-01-25600.csv'
path = "./data/predict"
maxFeature = []
minFeature = []
frequency = 4096


# coef = 0.71


# 读取预定数据
def readAllFiles():
    verticalData = []
    # 遍历文件夹
    for index in range(533):
        position = path + '/' + str(index + 1) + ".csv"
        test = np.loadtxt(position, dtype=np.str)
        test = np.delete(test, 0, 0)
        for data in test:
            # 将水平传感器状态数据转化为浮点型
            verticalData.append(float(data.split(",")[0]))
    return verticalData


# 数据同化
def origin2train(old, new):
    oldMean = np.mean(old)
    newMean = np.mean(new[round(np.size(new) / 2):np.size(new):1])
    old = old + (newMean - oldMean)
    return old


class Normalization:
    # 训练集归一化（第二版本）
    def trainNormalizedV2(self, trainArr):
        # 记录最大最小值
        print("训练集归一化开始")
        global maxFeature, minFeature
        maxFeature = np.array(trainArr).max(axis=0)
        minFeature = np.array(trainArr).min(axis=0)
        # 矩阵归一化
        min_max_scaler = preprocessing.MinMaxScaler()
        normalMatrix = min_max_scaler.fit_transform(trainArr)
        return normalMatrix

    # 测试集归一化处理（第二个版本）
    def testNormalizedV2(self, testArr):
        print("测试集归一化开始")
        testNormal = np.transpose(testArr)
        #  归一化处理
        for dimension in range(len(testNormal)):
            maxTestFeature = maxFeature[dimension]
            minTestFeature = minFeature[dimension]
            for index in range(len(testNormal[dimension])):
                testNormal[dimension][index] = (testNormal[dimension][index] - minTestFeature) / (
                        maxTestFeature - minTestFeature)
        return np.transpose(testNormal) / self.zoom_factor + self.translation_factor

    # 输出预测结果到文件
    def toCSV(result, path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(result)

    def __init__(self, filePath, zoomFactor, translationFactor):
        self.X_train = []
        self.X_predict = []
        self.filePath = filePath
        self.zoom_factor = zoomFactor
        self.translation_factor = translationFactor

        # 获取训练集数据
        allPreData = pd.read_csv('./data/originTrainData.csv').values.flatten().tolist()
        allPreDataArr = cut(allPreData, frequency)
        for index in allPreDataArr:
            self.X_train.extend(getFeature(np.array([index])))
        self.X_train = self.trainNormalizedV2(self.X_train)

        # 获取测试集数据
        txtFile = np.transpose(np.loadtxt(self.filePath, dtype=np.float128))
        self.X_predict.extend(getFeature(np.array([txtFile])))
        # X_predict = origin2train(X_predict, allPreData)
        self.X_predict = self.testNormalizedV2(self.X_predict)

    # # 根据txt文件生成预测集特征向量
    # X_train = []
    # # allPreData = readAllFiles()
    # # toCSV(allPreData, "./data/originTrainData.csv")
    # allPreData = pd.read_csv('./data/originTrainData.csv').values.flatten().tolist()
    # allPreDataArr = cut(allPreData, frequency)
    # for index in allPreDataArr:
    #     X_train.extend(getFeature(np.array([index])))
    # X_train = trainNormalizedV2(X_train)
    #
    # X_predict = []
    # txtFile = np.transpose(np.loadtxt(predictPath, dtype=np.float128))
    # X_predict.extend(getFeature(np.array([txtFile])))
    # # X_predict = origin2train(X_predict, allPreData)
    # X_predict = testNormalizedV2(X_predict)
