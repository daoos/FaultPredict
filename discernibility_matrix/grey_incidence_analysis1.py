#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np

# s表示信息系统
S = np.loadtxt('data.txt')
# X对应分量的均值
S_mean = np.mean(S, axis=1)
# 对X和X0中的每个分量进行均值化处理
S_process = np.transpose(np.transpose(S) / S_mean)
# 将第一行作为参考数列，X0_process为均值化处理后的参考数列
X0_process = S_process[0]
# 将其他作为比较数列，X_process为均值化处理后的比较数列
X_process = S_process[1:]
# 求比较数列与参考数列对应分量之差的绝对值序列delta(k)
delta = np.abs(X_process - X0_process)
# 绝对值序列的最大值M和最小值m
M = np.max(delta)
m = np.min(delta)
# 关联系数 0.5为分辨系数
gama = (m + 0.5 * M) / (delta + 0.5 * M)# 关联度
GAMA = np.mean(gama, axis=1)
print(GAMA)
# np.delete(S, GAMA.argsort()[:3] + 1, axis=0)表示删除灰色关联度较小的前3个样本数据，
# 并将其保存到文件中
np.savetxt('data_process.txt', np.delete(S, GAMA.argsort()[:3] + 1, axis=0),fmt='%.2f')
