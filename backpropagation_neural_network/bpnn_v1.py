#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import sklearn.preprocessing as preprocessing


class NeuralNetwork:
    def __init__(self, layers, input_array, learning_rate=0.05, activation='tanh'):
        """
        :param layers: 一个list，list中包含神经网络各层units的个数，至少为3层
        :param input_array: 训练样本数据
        :param learn_step: 学习步长，默认情况下为0.05
        :param activation: 激活函数，通常为"sigmoid","tanh","ReLU"，默认为"tanh"
        """
        self.layers = layers
        # 根据传递的参数选择不同的激活函数
        self.activation = self.activation_function(activation=activation)
        self.derivative = self.activation_derivative(activation=activation)
        # 偏置
        self.biases = [np.random.randn(y) for y in self.layers[1:]]
        # 权重
        self.weights = [np.random.randn(y, x) / np.sqrt(2.0 / x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        # 学习步长
        self.learning_rate = learning_rate

    def activation_function(self, activation, x):
        """
        根据激活函数名称选择激活函数
        :param activation: 激活函数名称
        :param x:
        :return:
        """
        if activation == 'tanh':
            return np.tanh(x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'ReLU':
            return np.maximum(0, x)

    def activation_derivative(self, activation, x):
        """
        根据激活函数导数
        :param activation: 激活函数名称
        :param x:
        :return:
        """
        if activation == 'tanh':
            return 1.0 - x ** 2
        elif activation == 'sigmoid':
            return x * (1 - x)
        elif activation == 'ReLU':
            deriv = 0
            if x > 0:
                deriv = 1
            return deriv

    def data_normalization(self, input_array):
        """
        对输入的数据进行归一化处理，将每个维度的数据归一化到[-1,1]
        :param input_array:样本数据
        :return:归一化之后的样本
        """
        return preprocessing.minmax_scale(input_array, feature_range=(-1, 1))

    def forward(self, input_data):
        """
        前向传播
        :param input_data: 输入层的样本集
        :return:
        """
        self.layers_values = []  # 每次循环都要初始化self.layers 为空
        # 每一层以上一层的值为输入，值= 上层节点的值 * 系数
        self.layers_values.append(input_array)  # 首层节点的值已知
        for i in range(len(self.layers) - 1):
            # 计算w^T*x + b，利用权值计算输出
            laywer_raw = np.dot(self.weights[i], np.transpose(self.layers_values[i])) + self.biases[i]
            # 为该层每个节点代入激活函数求值
            laywer_value = self.activation(laywer_raw)
            # 将该层输出值，存入列表,这样子便得到了一次正向传播后所有的层的所有节点的值
            self.layers_values.append(laywer_value)
        return laywer_value[-1]
