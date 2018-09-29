# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def linear_layer(value, output_dim, name='linear_connected'):
    with tf.variable_scope(name):
        try:
            weights = tf.get_variable('weights', [int(value.get_shape()[1]), output_dim],
                                      initializer = tf.truncated_normal_initializer(stddev =0.02))
            biases = tf.get_variable('biases', [output_dim], initializer = tf.constant_initializer(0.0))
        except ValueError:
            tf.get_variable_scope().reuse_variables()
            weights = tf.get_variable('weights', [int(value.get_shape()[1]), output_dim],
                                      initializer=tf.truncated_normal_initializer(stddev = 0.02))
            biases = tf.get_variable('biases', [output_dim], initializer = tf.constant_initializer(0.0))
        return tf.matmul(value, weights) + biases
    pass


# 卷积层
def conv_2d(value, output_dim, k_h=5, k_w=5, strides=[1,1,1,1], name="conv_2d"):
    with tf.variable_scope(name):
        try:
            weights = tf.get_variable('weights', [k_h, k_w, int(value.get_shape()[-1]), output_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            biases = tf.get_variable('biases', [output_dim], initializer = tf.constant_initializer(0.0))
        except ValueError:
            tf.get_variable_scope().reuse_variables()
            weights = tf.get_variable('weights', [k_h, k_w, int(value.get_shape()[-1]), output_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv_2d(value, weights, strides=strides, padding="SAME")
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


# 反卷积层
def deconv_2d(value, output_shape, k_h=5, k_w=5, strides=[1,1,1,1], name="deconv_2d"):
    with tf.variable_scope(name):
        try:
            weights = tf.get_variable('weights',
                [k_h, k_w, output_shape[-1], int(value.get_shape()[-1])],
                initializer = tf.truncated_normal_initializer(stddev = 0.02))
            biases = tf.get_variable('biases',
                [output_shape[-1]], initializer = tf.constant_initializer(0.0))
        except ValueError:
            tf.get_variable_scope().reuse_variables()
            weights = tf.get_variable('weights', [k_h, k_w, output_shape[-1], int(value.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.conv_2d_transpose(value, weights, output_shape, strides=strides)
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv


# 把约束条件串联到 feature map
def conv_cond_concat(value, cond, name='concat'):

    # 把张量的维度形状转化成 Python 的 list
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()

    # 在第三个维度上（feature map 维度上）把条件和输入串联起来，
    # 条件会被预先设为四维张量的形式，假设输入为 [64, 32, 32, 32] 维的张量，
    # 条件为 [64, 32, 32, 10] 维的张量，那么输出就是一个 [64, 32, 32, 42] 维张量
    with tf.variable_scope(name):
        return tf.concat([value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])], 3, name=name)
    pass


# Batch Normalization 层
def batch_norm_layer(value, is_train=True, name='batch_norm'):
    with tf.variable_scope(name) as scope:
        if is_train:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True, is_training=is_train,
                              reuse=tf.AUTO_REUSE, updates_collections=None, scope=scope)
        else :
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True, is_training=is_train,
                              reuse=tf.AUTO_REUSE, updates_collections=None, scope=scope)
        pass
    pass


# Leaky-Relu 层
def Leaky_Relu(x, leak=0.2, name='Leaky_Relu'):
    with tf.variable_scope(name):
        return tf.maximum(x, x*leak, name=name)
    pass
