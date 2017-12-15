#!/usr/bin/python
# -*- coding: utf-8 -*-
# 采用于启发式的邻域近似条件熵进行属性约简
import numpy as np
from builtins import print
from sklearn import preprocessing
from scipy.spatial.distance import pdist
from functools import reduce

from sympy.core.tests.test_operations import join


class AttributeReduction:
    def __init__(self, U, C, D):
        """
        初始化邻域决策系统
        :param U:对象的有限集合,称为论域
        :param C:条件属性集合
        :param D:决策属性集合
        """
        # 论域
        self.universe = U
        # 条件属性
        self.conditional_attr = C
        # 决策属性集合
        self.decision_attr = D
        # 邻域决策系统的一个约简
        self.reduce = {}

    def decision_attr_divide(self):
        """
        论域U 在决策属性 D 上的划分
        :return: 决策属性的划分（集合）
        """
        # 划分
        divide_set = list()
        attr = np.unique(self.universe[:, -1])
        for i in attr:
            divide_set.append(np.argwhere(self.universe[:, -1] == i).flatten())
        return divide_set

    # def get_sample_neighborhood(self, _lambda):
    #     sigma = np.std(self.universe, axis=0) / _lambda
    #     sample_neighborhood = np.zeros(self.universe.shape, dtype=set)
    #     for i in np.arange(len(self.universe)):
    #         D_array = np.abs(self.universe[i] - self.universe) - sigma
    #         for j in np.arange(D_array.shape[1]):
    #             sample_neighborhood[i, j] = set(np.argwhere(D_array[:, j] < 0).flatten())
    #     return sample_neighborhood

    def get_NAH(self, relative_attr_set, attr_set, sigma):
        """
        属性attr_set相对于relative_attr_set的邻域近似条件熵
        :param relative_attr_set: 决策属性
        :param attr_set: 条件属性子集
        :param sigma: 邻域阈值
        :return:邻域近似条件熵
        """
        # 对象在条件属性子集B下的邻域
        neighbourhood_B = dict()
        # 对象在决策属性D下的邻域
        neighbourhood_D = dict()
        # 判断对象是否为正域对象标记
        positive_field_flag = True
        # 判断NAH(D|C)是否为最大值标记
        max_flag = True
        if not attr_set:
            return len(self.universe) * np.log2(len(self.universe))
        # 论域中每个样本在属性C上的sigma邻域的相关计算
        for index in np.arange(len(self.universe)):
            # reduce功能 将属于属性C的每个属性得到的邻域求并集
            neighbourhood_B[index] = set(np.argwhere(np.sqrt(
                np.sum(np.square(self.universe[index, attr_set] - self.universe[:, attr_set]),
                       axis=1)) < sigma).flatten())
            neighbourhood_D[index] = set(np.argwhere(np.sqrt(
                np.sum(np.square(self.universe[index, relative_attr_set] - self.universe[:, relative_attr_set]),
                       axis=1)) < sigma).flatten())
            # 判断x是否是y的子集
            if neighbourhood_B[index].difference(neighbourhood_D[index]):
                positive_field_flag = False
            if not len(neighbourhood_B[index]) is len(self.universe) or not len(neighbourhood_D[index]) is 1:
                max_flag = False
        # 判断对象是否都为正域对象
        if positive_field_flag:
            NAH = 0
        elif max_flag:
            NAH = len(self.universe) * np.log2(len(self.universe))
        else:
            # relative_attr_set相对于attr_set的邻域近似条件熵
            conditional_entropy = 0
            # 在决策属性上的划分
            divide_set = self.decision_attr_divide()
            for y in divide_set:
                factor1 = np.log2(2 - self.approximation_quality(attr_set, set(y), sigma))
                factor2 = 0
                for i in np.arange(len(self.universe)):
                    if len(neighbourhood_B[i] & set(y)) != 0:
                        a = len(neighbourhood_B[i] & set(y)) / len(self.universe)
                        factor2 += a * np.log2(a)
                conditional_entropy += factor1 * factor2
            NAH = -conditional_entropy
        return NAH

    def get_index(self, current_set, reference_set):
        """
        :param current_set: 当前集合
        :param reference_set: 参考集合
        :return: 当前集合中的元素在参考集合中的索引
        """
        return [list(reference_set).index(a) for a in current_set]

    def approximation_quality(self, B, X, sigma):
        """
        X 在邻域关系 N_B 下的邻域近似精度
        :param B: 条件属性子集
        :param X:决策属性等价类子集
        :param sigma: 样本邻域阈值
        """
        # 上近似
        up_neighborhood = set()
        # 下近似
        down_neighborhood = set()
        for index in np.arange(len(self.universe)):
            neigh = set(np.argwhere(
                np.sqrt(np.sum(np.square(self.universe[index, B] - self.universe[:, B]), axis=1)) < sigma).flatten())
            # 判断第index个对象的邻域neigh是否是X的子集
            if not neigh.difference(X):
                down_neighborhood.add(index)
            # 判断第index个对象的邻域neigh与X是否有交集
            if neigh & X:
                up_neighborhood.add(index)
        return len(down_neighborhood) / len(up_neighborhood)


if __name__ == '__main__':
    # 加载决策系统对象U
    DS = np.loadtxt('data_process.txt')
    # 规范化处理
    U = preprocessing.minmax_scale(DS)
    A = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'D']
    # 条件属性
    C = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    # 决策属性
    D = ['D']
    ar = AttributeReduction(U, C, D)
    # 得到决策属性的划分
    divide_set = ar.decision_attr_divide()
    sigma = 0.3
    # 邻域近似条件熵
    NAH_D_C = ar.get_NAH(ar.get_index(D, A), ar.get_index(C, A), sigma)
    C_array = ar.get_index(C, A)
    D_array = ar.get_index(D, A)
    # 邻域决策系统的约简
    B = set()
    for i in C_array:
        temp = C_array.copy()
        temp.remove(i)
        SIG_I_C_D = ar.get_NAH(D_array, temp, sigma) - NAH_D_C
        if SIG_I_C_D > 0:
            B.add(i)
    print(B)
    NAH_D_B = ar.get_NAH(D_array, list(B), sigma)
    while NAH_D_C != NAH_D_B:
        R = set(C_array) - B
        SIG_I_B_D = dict()
        for a in R:
            SIG_I_B_D[a] = NAH_D_B - ar.get_NAH(D_array, list(B.union({a})), sigma)
        # sort函数按照值进行s排序，通过下标-1选出值最大的键值对，通过下标0拿到键值对的键
        print(SIG_I_B_D)
        a_i = sorted(SIG_I_B_D.items(), key=lambda item: item[1])[-1][0]
        R.remove(a_i)
        B = B.union({a_i})
        NAH_D_B = ar.get_NAH(D_array, list(B), sigma)
    # 加上最后一行决策属性
    indexs = list(B) + ar.get_index(D, A)
    print(indexs)
    np.savetxt('data.txt', DS[:, indexs], fmt='%0.2f')
