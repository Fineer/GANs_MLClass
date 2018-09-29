# -*- coding: utf-8 -*-
import scipy.misc
import numpy as np
import tensorflow as tf
import os
from read_data import *
from operations import *
from model import *

BATCH_SIZE = 64


# 保存图片的函数
def save_images(images, size, path):
    # 图片归一化，主要用于生成器输出是 tanh 形式的归一化
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]

    # 产生一个大画布，用来保存生成的 batch_size 个图像
    merge_img = np.zeros((h * size[0], w * size[1], 3))

    # 循环使得画布特定地方值为某一幅图像的值
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    # 保存画布
    return scipy.misc.imsave(path, merge_img)


def train():
    # 读取数据
    X, Y = read_data()

    # 设置 global_step ，用来记录训练过程中的 step
    global_step = tf.Variable(0, name='global_step', trainable=True)

    # 放置三个 placeholder，y 表示约束条件，images 表示送入判别器的图片，
    # z 表示随机噪声
    y = tf.placeholder(tf.int32, [BATCH_SIZE], name='y')
    _y = tf.one_hot(y, depth=10, on_value=None, off_value=None, axis=None, dtype=None, name='one_hot')
    z = tf.placeholder(tf.float32, [BATCH_SIZE, 100], name='z')
    images = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1], name='images')

    # 由生成器生成图像 G
    G = generator(z, _y)
    # 真实图像送入判别器
    D = discriminator(images, _y)
    # 生成图像送入判别器
    _D = discriminator(G, _y)

    # 计算 sigmoid 交叉熵的损失
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D, labels=tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_D, labels=tf.zeros_like(_D)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_D, labels=tf.ones_like(_D)))
    d_loss = d_loss_real + d_loss_fake

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars, global_step=global_step)
        g_optim = tf.train.AdamOptimizer(0.0002, beta2=0.5).minimize(g_loss, var_list=g_vars, global_step=global_step)

    # tensorborad
    train_dir = 'logs'
    z_sum = tf.summary.histogram("z", z)
    d_sum = tf.summary.histogram("d", D)
    d__sum = tf.summary.histogram("d_", _D)
    g_sum = tf.summary.histogram("g", G)

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)

    g_sum = tf.summary.merge([z_sum, d__sum, g_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])

    # initial
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    writer = tf.summary.FileWriter(train_dir, sess.graph)

    # save
    saver = tf.train.Saver()
    check_path = "save/model.ckpt"

    # sample
    sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
    sample_labels = Y[0:BATCH_SIZE]

    # make sample
    sample = sampler(z, _y)

    # run
    sess.run(init)

    # 循环25个epoch的训练网络
    for epoch in range(25):
        batch_idx = int(70000 / 64)
        for idx in range(batch_idx):
            batch_images = X[idx * 64:(idx + 1) * 64]
            batch_labels = Y[idx * 64:(idx + 1) * 64]
            batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            # 更新 D 的参数
            _, summary_str = sess.run([d_optim, d_sum],
                                      feed_dict={images: batch_images,
                                                 z: batch_z,
                                                 y: batch_labels})
            writer.add_summary(summary_str, idx + 1)

            # 更新 G 的参数
            _, summary_str = sess.run([g_optim, g_sum],
                                      feed_dict={images: batch_images,
                                                 z: batch_z,
                                                 y: batch_labels})
            writer.add_summary(summary_str, idx + 1)

            d_loss1 = d_loss_fake.eval({z: batch_z, y: batch_labels})
            d_loss2 = d_loss_real.eval({images: batch_images, y: batch_labels})
            D_loss = d_loss1 + d_loss2
            G_loss = g_loss.eval({z: batch_z, y: batch_labels})

            # 每20个batch输出一次损失
            if idx % 20 == 0:
                s = str("Epoch %d [%4d/%4d] d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idx, D_loss, G_loss))
                with open('losses_of_epoches.txt', 'a+') as f:
                    f.write(s+'\n')
                print(s)

            # 每 100batch 输出一张图片
            if idx % 100 == 0:
                sap = sess.run(sample, feed_dict={z: sample_z, y: sample_labels})
                samples_path = 'sample\\'
                save_images(sap, [8, 8], samples_path + 'test_%d_epoch_%d.png' % (epoch, idx))

            # 每 500 次迭代保存一次模型
            if idx % 500 == 0:
                saver.save(sess, check_path, global_step=idx + 1)
    sess.close()


if __name__ == '__main__':
    train()
