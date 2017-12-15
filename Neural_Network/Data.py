#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


class Data:
    def __init__(self, input_array, batch_size=10):  # 数据所在的文件名name和batch中图片的数量batch_size

        # 输入X，X为矩阵
        self.input = input_array[:-1, :]
        # 期望正确的输出y
        self.output = input_array[-1, :]
        # 样本集中样本的个数
        self.size = self.input.shape[1]
        # 将样本集分组进行操作，每组样本集的个数应不超过batch_size
        self.batch_size = batch_size
        self.pos = 0  # pos用来记录数据读取的位置

    def forward(self):
        pos = self.pos
        bat = self.batch_size
        l = self.size
        # 已经是最后一个batch时，返回剩余的数据，并设置pos为开始位置0
        if pos + bat >= self.size:
            ret = (self.input[:, pos:self.size], self.output[:, pos:self.size])
            self.pos = 0
            index = np.arange(self.size)
            np.random.shuffle(index)  # 将训练数据打乱
            self.input = self.input[:, index]
            self.output = self.output[:, index]
        else:  # 不是最后一个batch, pos直接加上batch_size
            ret = (self.input[:, pos:pos + bat], self.output[:, pos:pos + bat])
            self.pos += self.batch_size
        return ret, self.pos  # 返回的pos为0时代表一个epoch已经结束

    def backward(self, d):  # 数据层无backward操作
        pass
