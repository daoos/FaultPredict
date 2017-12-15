#!/usr/bin/python
# -*- coding: utf-8 -*-
# encoding=utf-8
from Neural_Network.Activation import *
from Neural_Network.FullyConnect import *
from Neural_Network.QuadraticLoss import *
from Neural_Network.Data import *
from sklearn import preprocessing as pp


def main():
    # 训练集个数
    data = np.asmatrix(pp.minmax_scale(np.loadtxt('data.txt'))).T
    train_set = data[:, :-1]
    test_set = data[:, -1]
    batch_size = 10
    datalayer1 = Data(train_set, batch_size)  # 用于训练，batch_size设置为10
    datalayer2 = Data(test_set, batch_size)  # 用于验证，所以设置batch_size为10,一次性计算所有的样例
    # 神经网络中间层
    inner_layers = []
    inner_layers.append(FullyConnect(4, 15))
    inner_layers.append(Activation(activation='relu'))
    inner_layers.append(FullyConnect(15, 1))
    inner_layers.append(Activation(activation='sigmoid'))
    # 损失层
    losslayer = QuadraticLoss()

    for layer in inner_layers:
        layer.learning_rate = 0.8  # 为所有中间层设置学习速率

    epochs = 20
    for i in range(epochs):
        print('epochs:', i)
        losssum = 0
        iters = 0
        while True:
            data, pos = datalayer1.forward()  # 从数据层取出数据
            x, label = data
            for layer in inner_layers:  # 前向计算
                x = layer.forward(x)

            loss = losslayer.forward(x, label)  # 调用损失层forward函数计算损失函数值
            losssum += loss
            iters += 1
            d = losslayer.backward()  # 调用损失层backward函数层计算将要反向传播的梯度

            for layer in inner_layers[::-1]:  # 反向传播
                d = layer.backward(d)

            if pos == 0:  # 一个epoch完成后进行准确率测试
                data, _ = datalayer2.forward()
                x, label = data
                for layer in inner_layers:
                    x = layer.forward(x)
                print('输出值：', x, '期望值：', label)
                print('loss:', losssum / iters)
                break


if __name__ == '__main__':
    main()
