import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

start = 300
extendLen = 500

Y = np.transpose(np.loadtxt("/Users/Johnny/Desktop/parameter.txt", dtype=np.float128))
X = np.linspace(1, np.size(Y) + extendLen, np.size(Y) + extendLen)
print("X length:", np.size(X))
# 切割新数字，从0到start，求取均值
startArr = Y[slice(0, start, 1)]
endArr = Y[slice(start, np.size(Y), 1)]
extendY = np.array([])
for index in range(extendLen):
    temp = np.random.randint(np.size(startArr))
    extendY = np.append(extendY, startArr[temp])
Y = np.append(np.append(startArr, extendY), endArr)
print("Y length:", np.size(Y))

# 数据降噪
n = 15
b = [1.0 / n] * n
a = 1
yy = lfilter(b, a, Y)
plt.plot(X, yy)
plt.show()