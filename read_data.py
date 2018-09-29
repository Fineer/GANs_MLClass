# -*- coding: utf-8 -*-
import os
import numpy as np


def read_data():

    # 数据目录
    data_dir = "minist"

    # 打开训练数据
    f = open(os.path.join(data_dir,"train-images.idx3-ubyte"))

    # 转化成 numpy 数组
    data_array = np.fromfile(file=f, dtype=np.uint8)

    # 根据 mnist 官网描述的数据格式，图像像素从 16 字节开始
    train_x = data_array[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    # 训练 label
    f = open(os.path.join(data_dir,"train-labels.idx1-ubyte"))
    data_array = np.fromfile(file = f, dtype = np.uint8)
    train_y = data_array[8:].reshape(60000).astype(np.float)

    # 测试数据
    f = open(os.path.join(data_dir,"t10k-images.idx3-ubyte"))
    data_array = np.fromfile(file=f, dtype=np.uint8)
    test_x = data_array[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    # 测试 label
    f = open(os.path.join(data_dir,"t10k-labels.idx1-ubyte"))
    data_array = np.fromfile(file=f, dtype=np.uint8)
    test_y = data_array[8:].reshape(10000).astype(np.float)

    # 把训练和测试两部分数据合并
    X = np.concatenate((train_x, test_x), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    # 打乱排序
    seed = 233
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X/255, y
