#! /usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 经过灰色关联分析后的信息系统S
# 供电电压220v，
S = np.loadtxt('data_process.txt')
for i in np.arange(S[:, :-1].shape[1]):
    fac, bins = pd.qcut(S[:, i], [0, 0.5, 1.0], [0, 1], retbins=True)
    S[:, i] = fac
    print(bins)
S = np.array(S, dtype=int)
print(S)
# 数组保存到文件
np.savetxt('data_efb_process.txt', S, fmt='%d')
