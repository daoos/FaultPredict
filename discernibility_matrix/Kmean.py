#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

detetion_data = np.loadtxt('data.txt')
data = detetion_data[:, :-1]
L2 = np.array([np.sum(np.square(i - data), axis=1) for i in data])
# np.savetxt('distance.txt', L2, fmt='%.4f')
center1 = data[int(np.argmax(L2) / 17)]
center2 = data[np.argmax(L2) % 17]
class_1_list = set()
class_2_list = set()
class_1_list.add(int(np.argmax(L2) / 17))
class_2_list.add(np.argmax(L2) % 17)
remark = 1
while remark >= 0.01:
    # 判断根据样本到两个类中心点的距离聚类
    for index in np.arange(len(data)):
        distance1 = np.sum(np.square(data[index] - center1))
        distance2 = np.sum(np.square(data[index] - center2))
        # print(distance1, distance2)
        if distance1 < distance2:
            class_1_list.add(index)
        else:
            class_2_list.add(index)
    print('----------------------------')
    print(class_1_list)
    print(class_2_list)
    new_center1 = np.mean(data[list(class_1_list)], axis=0)
    new_center2 = np.mean(data[list(class_2_list)], axis=0)
    remark = np.sum(np.square(new_center1 - center1)) + np.sum(np.square(new_center2 - center2))
    center1 = new_center1
    center2 = new_center2
print(detetion_data[list(class_1_list)][:, -1])
print(detetion_data[list(class_2_list)][:, -1])
