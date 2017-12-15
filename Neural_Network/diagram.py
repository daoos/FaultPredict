#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

precision_set = np.load('precision_set.npy').reshape(-1, 1)
real_set = np.load('real_set.npy').reshape(-1, 1)
# fault_data = np.column_stack((precision_set, real_set))
# np.savetxt('fault_data1.txt',fault_data,fmt='%.4f')
print(np.sum(np.abs(precision_set - real_set) < 0.5))
font_set = FontProperties(fname="C:\Windows\Fonts\simkai.ttf", size=15)
plt.scatter([i for i in range(len(precision_set))], precision_set, label='样本预测值')
plt.scatter([i for i in range(len(real_set))], real_set, label='样本真实值')
plt.plot([i for i in range(len(precision_set))], precision_set, label='样本预测值')
plt.plot([i for i in range(len(real_set))], real_set, label='样本真实值')
plt.xlabel('样本序号', fontproperties=font_set)
plt.ylabel('故障情况', fontproperties=font_set)
plt.title('故障预测', fontproperties=font_set)
plt.ylim(-0.5, 1.2)
plt.legend(prop=font_set)
plt.show()
