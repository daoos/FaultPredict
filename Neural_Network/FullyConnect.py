#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


class FullyConnect:
    def __init__(self, input_size, output_size):
        """
        初始化全连接层
        :param input_size: 输入层的长度
        :param output_size: 输出层的长度
        """
        # 使用随机数初始化该层权重
        self.weights = np.random.randn(output_size, input_size).astype(np.double) / np.sqrt(2 / input_size)
        # 使用随机数初始化该层偏置量
        self.bias = np.random.randn(output_size, 1).astype(np.double)
        # 先将学习速率初始化为0.05，也可以统一设定
        self.learning_rate = 0.05
        # 输入
        self.input = None
        # 输出
        self.output = None

    def forward(self, input_array):
        """
        :param input_array:输入的样本集
        :return: 前向传播的结果
        """
        self.input = input_array  # 把中间结果保存下来，以备反向传播时使用
        self.output = np.dot(self.weights, input_array) + self.bias  # 计算全连接层的输出
        return self.output  # 将这一层计算的结果向前传递

    def backward(self, dz):
        """
        反向传播
        :param dz: 通过反向传播得到的导数
        :return:
        """
        # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        assert (dz.shape[0] == (self.weights.shape[0]))
        self.delta_weight = np.dot(dz, self.input.T) / self.input.shape[1]
        self.delta_bias = np.sum(dz, axis=1, keepdims=True) / self.input.shape[1]
        self.delta_x = np.matmul(self.weights.T, dz)

        # 更新权重和偏置量
        self.weights -= self.learning_rate * self.delta_weight
        self.bias -= self.learning_rate * self.delta_bias
        return self.delta_x  # 反向传播梯度da
