#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

data = np.loadtxt('data.txt')[:, :-1]
L2 = np.array([np.sum(np.square(i - data), axis=1) for i in data])
np.savetxt('distance.txt', L2, fmt='%.4f')
center1 = int(np.argmax(L2) / 17)
center2 = np.argmax(L2) % 17
class_1_list = set()
class_2_list = set()
class_1_list.add(center1)
class_2_list.add(center2)
for index in np.arange(len(data)):
    distance1 =  np.square(data[index] - center1)
