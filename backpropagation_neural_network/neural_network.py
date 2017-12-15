#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.设定输入层、隐藏层和输出层的node数目
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights，初始化权重和学习速率
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        # 隐藏层的激励函数为sigmoid函数，Activation function is the sigmoid function
        self.activation_function = (lambda x: 1 / (1 + np.exp(-x)))

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T  # 输入向量的shape为 [feature_diemension, 1]
        targets = np.array(targets_list, ndmin=2).T

        # 向前传播，Forward pass
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # 输出层，输出层的激励函数就是 y = x
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        ### 反向传播 Backward pass，使用梯度下降对权重进行更新 ###

        # 输出误差
        # Output layer error is the difference between desired target and actual output.
        output_errors = (targets_list - final_outputs)

        # 反向传播误差 Backpropagated error
        # errors propagated to the hidden layer
        hidden_errors = np.dot(output_errors, self.weights_hidden_to_output) * (hidden_outputs * (1 - hidden_outputs)).T

        # 更新权重 Update the weights
        # 更新隐藏层与输出层之间的权重 update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += output_errors * hidden_outputs.T * self.lr
        # 更新输入层与隐藏层之间的权重 update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += (inputs * hidden_errors * self.lr).T

    # 进行预测
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### 实现向前传播 Implement the forward pass here ####
        # 隐藏层 Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # 输出层 Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs
