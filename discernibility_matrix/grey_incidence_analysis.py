#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing as prep

# s表示信息系统
S = np.loadtxt('data.txt')
# 无量纲化处理
temp = prep.scale(S[:, :-1], axis=1)
# 灰色关联分析当中的比较数列,去除最后一列的决策属性
X = temp[1:, :-1]
# X0为灰色关联分析当中的参考数列,去除最后一列的决策属性
X0 = temp[0, :-1]
# X对应分量的均值
X_mean = np.mean(X, axis=1)
X0_mean = np.mean(X0)
# 对X和X0中的每个分量进行均值化处理
X_process = np.transpose(np.transpose(X) / X_mean)
X0_process = X0 / X0_mean
# 求比较数列与参考数列对应分量之差的绝对值序列delta(k)
delta = np.abs(X_process - X0_process)
# 绝对值序列的最大值M和最小值m
M = np.max(delta)
m = np.min(delta)
# # 关联系数 0.5为分辨系数
gama = (m + 0.5 * M) / (delta + 0.5 * M)  # 关联度
GAMA = np.mean(gama, axis=1)
print(GAMA)
# np.delete(S, GAMA.argsort()[:3] + 1, axis=0)表示删除灰色关联度较小的前3个样本数据，
# 并将其保存到文件中
np.savetxt('data_process.txt', np.delete(S, GAMA.argsort()[:3] + 1, axis=0), fmt='%.2f')
