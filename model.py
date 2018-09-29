import tensorflow as tf
from operations import *

BATCH_SIZE = 64


# 生成器
def generator(z, y, train=True):
    # y 是一个 [BATCH_SIZE, 10] 维的向量，把 y 转成四维张量
    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 10], name='g_yb')
    # 把 y 作为约束条件和 z 拼接起来
    z_y = tf.concat([z, y], 1, name='g_z_concat_y')

    linear1 = linear_layer(z_y, 1024, name='g_linear_layer1')
    bn1 = tf.nn.relu(batch_norm_layer(linear1, is_train=True, name='g_bn1'))

    bn1_y = tf.concat([bn1, y], 1, name='g_bn1_concat_y')
    linear2 = linear_layer(bn1_y, 128 * 49, name='g_linear_layer2')
    bn2 = tf.nn.relu(batch_norm_layer(linear2, is_train=True, name='g_bn2'))
    bn2_re = tf.reshape(bn2, [BATCH_SIZE, 7, 7, 128], name='g_bn2_reshape')

    bn2_yb = conv_cond_concat(bn2_re, yb, name='g_bn2_concat_yb')
    deconv1 = deconv_2d(bn2_yb, [BATCH_SIZE, 14, 14, 128], strides=[1, 2, 2, 1], name='g_deconv1')
    bn3 = tf.nn.relu(batch_norm_layer(deconv1, is_train=True, name='g_bn3'))

    bn3_yb = conv_cond_concat(bn3, yb, name='g_bn3_concat_yb')
    deconv2 = deconv_2d(bn3_yb, [BATCH_SIZE, 28, 28, 1], strides=[1, 2, 2, 1], name='g_deconv2')
    return tf.nn.sigmoid(deconv2)


# 判别器
def discriminator(image, y, reuse=False):
    # 因为真实数据和生成数据都要经过判别器，所以需要指定 reuse 是否可用
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # 同生成器一样，判别器也需要把约束条件串联进来
    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 10], name='d_yb')
    image_yb = conv_cond_concat(image, yb, name='d_image_concat_yb')
    conv1 = conv_2d(image_yb, 11, strides=[1, 2, 2, 1], name='d_conv1')
    lr1 = Leaky_Relu(conv1, name='d_Leaky_Relu1')

    lr1_yb = conv_cond_concat(lr1, yb, name='d_lr1_concat_yb')
    conv2 = conv_2d(lr1_yb, 74, strides=[1, 2, 2, 1], name='d_conv2')
    bn1 = batch_norm_layer(conv2, is_train=True, name='d_bn1')
    lr2 = Leaky_Relu(bn1, name='d_Leaky_Relu2')
    lr2_re = tf.reshape(lr2, [BATCH_SIZE, -1], name='d_lr2_reshape')

    lr2_y = tf.concat([lr2_re, y], 1, name='d_lr2_concat_y')
    linear1 = linear_layer(lr2_y, 1024, name='d_linear_layer1')
    bn2 = batch_norm_layer(linear1, is_train=True, name='d_bn2')
    lr3 = Leaky_Relu(bn2, name='d_Leaky_Relu3')

    lr3_y = tf.concat([lr3, y], 1, name='d_lr3_concat_y')
    linear2 = linear_layer(lr3_y, 1, name='d_linear_layer2')

    return linear2


# 定义训练过程中的采样函数
def sampler(z, y, train=True):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, train=train)
