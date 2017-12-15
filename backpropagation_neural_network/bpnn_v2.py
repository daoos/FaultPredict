#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import sys
from sklearn import preprocessing as pp


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - x ** 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return x * (1 - x)


# 数据归一化处理
def data_normalization(data):
    return pp.minmax_scale(data)


class BackPropagationNeuralNetwork:
    def __init__(self, layers, input_array, output, learn_step=0.1, activation='tanh'):
        """
        :param layers: 一个list，list中包含神经网络各层units的个数，至少为3层
        :param input_array: 训练样本数据
        :param output: 期望输出结果
        :param learn_step: 学习步长
        :param activation: 激活函数，通常为"logistic"或者"tanh"，默认为"tanh"
        """
        self.layers = layers
        # 根据传递的参数选择不同的激活函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # 偏置
        self.biases = [np.random.randn(y) for y in self.layers[1:]]
        # 权重除以x防止激活函数达到饱和状态
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        # 期望输出
        self.desired_outputs = output
        # 样本的训练集
        self.input_array = input_array
        # 如果数据归一化,存储两个最大特征值 与 最小特征值的array ,不归一化的话,不用考虑
        self.norm_max_array = []
        self.norm_min_array = []
        # 各层神经元节点的输出
        self.layers_values = []
        # 权值修正时的学习步长
        self.learn_step = learn_step
        # 整体错误率
        self.error_total = 0

    # 数据归一化处理
    def data_normalization(self, input_array):
        self.norm_max_array = np.max(input_array, axis=0)
        self.norm_min_array = np.min(input_array, axis=1)
        min_max_scaler = pp.MinMaxScaler(feature_range=(0, 1))
        return min_max_scaler.fit_transform(input_array)

    # 前向传播
    def forward(self, input_array):
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

    # 代价函数
    def input_error(self, output):
        return output * np.log(self.layers_values[-1]) + (1 - output) * np.log(1 - self.layers_values[-1])

    # 反向传播
    def back_propagation(self, desired_outputs):
        """
        :param desired_outputs: 期望输出
        :return:
        """
        # 输出层的输出误差
        error = self.activation_deriv(self.layers_values[-1]) * (desired_outputs - self.layers_values[-1])
        # 输出层误差对权值的导数
        deltas = [error]
        # 中间层误差对权值的导数
        for j in np.arange(len(self.layers_values) - 2, 0, -1):
            delta = self.activation_deriv(self.layers_values[j]) * np.dot(deltas[0], self.weights[j])
            deltas.insert(0, delta)
        # 修正权值
        for i in np.arange(len(self.weights)):
            self.weights[i] += self.learn_step * np.dot(deltas[i].reshape(-1, 1), self.layers_values[i].reshape(1, -1))
            self.biases[i] += self.learn_step * deltas[i]

    # 训练模型
    def train(self):
        self.error_total = 0
        self.predict_result = []
        # 标准化数据
        # input_array = self.data_normalization(self.input_array)
        for i in np.arange(len(input_array)):
            self.forward(input_array[i])
            self.back_propagation(self.desired_outputs[i])
            self.error_total += self.input_error(self.desired_outputs[i])

    # 测试样本
    def test(self, test_set):
        self.predict_result = []
        # 标准化测试集数据
        # data = self.data_normalization(test_set)
        data = test_set
        for i in np.arange(len(data)):
            self.forward(data[i])
            self.predict_result.append(self.layers_values[-1])


if __name__ == '__main__':
    layers = [4, 6, 6, 1]
    data = np.loadtxt('data.txt')
    # 将data中的数据随机打乱
    np.random.shuffle(data)
    # 保存打乱后的数据
    np.savetxt('data.txt', data, fmt='%.2f')
    # 对数据进行归一化处理
    data = data_normalization(data)
    # 训练集个数
    train_num = 13
    # 训练集
    input_array = data[:train_num, :-1]
    # 训练集的期望输出
    output_array = data[:train_num, -1]
    # 测试集
    test_input_set = data[train_num:, :-1]
    # 测试集的期望输出
    test_output_set = data[train_num:, -1]
    bpnn = BackPropagationNeuralNetwork(layers, input_array=input_array, output=output_array,activation='logistic')
    error = 1
    count = 0
    while count < 50000 and np.abs(error) > 0.1:
        count += 1
        bpnn.train()
        error = bpnn.error_total
        print(error)
    np.save('bpnn_weights.npy', bpnn.weights)
    np.save('bpnn_biases.npy', bpnn.biases)
    file_name = 'output.txt'
    sys.stdout = open(file_name, 'w+', encoding='utf-8')
    print('训练样本的期望输出：', output_array)
    predict_result = list()
    for unit in input_array:
        predict_result.append(bpnn.forward(unit))
    print('训练样本的预测值：', predict_result)
    print('训练执行次数：', count)
    # 预测
    bpnn.test(test_input_set)
    print('测试样本的期望输出：', test_output_set)
    print('测试样本的预测值：', bpnn.predict_result)
    sys.stdout.close()
