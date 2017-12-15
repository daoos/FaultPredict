#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from numpy.core.multiarray import format_longfloat


class Activation:
    """
    激活函数
    """

    def __init__(self, activation):  # 无参数，不需初始化
        # activation 激活函数
        # activation_deriv 激活函数的导数
        if str.lower(activation) == str.lower('tanh'):
            self.activation = self.tanh
            self.activation_deriv = self.tanh_deriv
        elif str.lower(activation) == str.lower('sigmoid'):
            self.activation = self.sigmoid
            self.activation_deriv = self.sigmoid_deriv
        elif str.lower(activation) == str.lower('ReLU'):
            self.activation = self.ReLU
            self.activation_deriv = self.ReLU_deriv

    def sigmoid(self, x):
        return (1.0 / (1.0 + np.exp(-x))).astype(np.double)

    def ReLU(self, x):
        return np.maximum(x, 0.0)

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid_deriv(self, x):
        return x * (1.0 - x)

    def ReLU_deriv(self, x):
        x[x > 0.0] = 1
        return x

    def tanh_deriv(self, x):
        return 1.0 - x ** 2

    def forward(self, z):
        self.input = np.array(z, dtype=np.double)
        self.output = self.activation(self.input)
        return self.output

    def backward(self, da):
        self.delta_x = da * self.activation_deriv(self.output)
        assert (self.delta_x.shape == self.input.shape)
        return self.delta_x  # 反向传递梯度
