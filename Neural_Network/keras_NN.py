#!/usr/bin/python
# -*- coding: utf-8 -*-
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import numpy as np
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt

# fix random seed for reproducibility
# 设置训练的次数
size = 10
precision_set = list()
real_set = list()
x = list()
# load pima indians dataset
data = np.asmatrix(pp.minmax_scale(np.loadtxt('data.txt')))
index = np.arange(data.shape[0])
for i in np.arange(20):
    np.random.shuffle(index)
    data = data[index, :]
    # 将数据集分成输入集(X)和输出集(Y)变量
    X_train = data[:-1, :-1]
    Y_train = data[:-1, -1]
    X_test = data[-1:, :-1]
    Y_test = data[-1:, -1]
    # 创建模型
    model = Sequential()
    model.add(Dense(30, input_dim=4, kernel_initializer="normal", activation='relu'))
    model.add(Dense(1, kernel_initializer="normal", activation='sigmoid'))
    sgd = optimizers.sgd(lr=0.03, momentum=0.9, decay=0.0, nesterov=False)
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy', 'binary_crossentropy'])
    # Fit the model
    model.fit(X_train, Y_train, epochs=4000, batch_size=16)
    # evaluate the model
    scores = model.evaluate(X_test, Y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # calculate predictions
    predictions = model.predict(X_test)
    print('预测值：', predictions, '\n真实值：', Y_test)
    precision_set.append(predictions[0][0])
    real_set.append(float(Y_test))
np.save('precision_set.npy', precision_set)
np.save('real_set.npy', real_set)
