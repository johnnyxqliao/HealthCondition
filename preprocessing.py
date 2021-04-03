# coding: utf-8
import numpy as np
import math
import pywt


# 样本切割函数
def cut(obj, sec):
    splitArr = []
    for i in range(0, len(obj), sec):
        splitArr.append(np.array(obj[i:i + sec]))
    return splitArr


# 读取excel文件
def readExcel(excelDir):
    global txtFile
    try:
        txtFile = np.transpose(np.loadtxt(excelDir, dtype=np.float128))
    except:
        print('没有找到对应的目标文件\n 请核对后重新运行镜像 \n')
    data = []
    frequency = 4096
    for index in txtFile:
        data.append(cut(index, frequency))
    return data


# 时域特征参数提取
def getTimeDomainFeature(self):
    preprocessResult = []
    for row in self:
        absArr = np.abs(row)
        remsValue = math.sqrt(np.sum([pow(x, 2) for x in row]) / row.size)
        cfValue = np.max(absArr) / remsValue
        sfValue = len(row) * remsValue / np.sum(np.abs(row))
        mfValue = getMF(absArr)
        snValue = getSN(row)
        ifValue = np.max(absArr) / np.mean(absArr)
        kfValue = getKF(row)
        energyFeature = getEnergyFeature(row)
        timeDomainFeature = [remsValue, cfValue, sfValue, ifValue, mfValue, kfValue, snValue]
        timeDomainFeature.extend(energyFeature)
        preprocessResult.append(timeDomainFeature)
    return preprocessResult


# 去均值
def getAvg(arr):
    avgArr = np.mean(arr, axis=1)
    for row in range(len(arr)):
        for index in range(len(arr[row])):
            arr[row][index] = arr[row][index] - avgArr[row]


# 计算mf
def getMF(arr):
    maxValue = np.max(arr)
    mf = pow(arr.size, 2) * maxValue / pow(np.sum([pow(x, 0.5) for x in arr]), 2)
    return mf


# 偏斜度
def getSN(arr):
    mean = np.mean(arr)  # 计算均值
    sn = np.mean((arr - mean) ** 3)
    return sn


# 峭度指标
def getKF(arr):
    mean = np.mean(arr)  # 计算均值
    std = np.std(arr)  # 计算标准差
    kf = np.mean([pow((x - mean) / std, 4) for x in arr])
    return kf


# 获取小波能量特征
def getEnergyFeature(arr):
    # 小波能量分解
    wp = pywt.WaveletPacket(data=arr, wavelet='db3', maxlevel=3)
    # 提取小波分解能量特征
    energy = []
    for i in [node.path for node in wp.get_level(3, 'freq')]:
        energy.append(np.linalg.norm(wp[i].data, ord=None))
    return energy / np.sum(energy)


# 获取信号特征
def getFeature(originArr):
    print("获取信号特征向量开始，入参为：", originArr)
    getAvg(originArr)
    featureArr = getTimeDomainFeature(np.array(originArr))
    return featureArr
