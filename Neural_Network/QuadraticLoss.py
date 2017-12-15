#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


class QuadraticLoss:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.input = np.array(x)
        self.label = np.array(label)
        sum_loss = np.nan_to_num(self.label * np.log(self.input) + (1 - self.label) * np.log(1 - self.input))
        self.loss = -np.sum(sum_loss, axis=1, keepdims=np.True_) / self.input.shape[1]
        return self.loss

    def backward(self):
        self.delta_x = (self.label - self.input) / (self.input * (1 - self.input))
        assert (self.delta_x.shape == self.input.shape)
        return self.delta_x
