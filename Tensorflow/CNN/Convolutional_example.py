# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/30 下午3:05'
import tensorflow as tf
'''
通过tf.get_variable 的方式创建过滤器的权重变量和偏置项变量。上面介绍了卷积层的参数个数和过滤器的尺寸、
深度以及当前层节点矩阵的深度有关，所以这里声明的参数变量时一个四维矩阵，前面两个维度代表了过滤器的尺寸，
第三个维度表示当前层的深度，第四个维度表示过滤器的深度。
'''
filter_weight = tf.get_variable(
    'weights', [5, 5, 3, 16],
    initializer=tf.truncated_normal_initializer(stddev=0.1))
'''
和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一层深度个不同的偏置项。
本样例中16为过滤器的深度，也是神经网络中下一层节点矩阵的深度。
'''
biases = tf.get_variable(
    'biases', [16], initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(
    input(), filter_weight, strides=[1, 1, 1, 1], padding='SAME')

bias = tf.nn.bias_add(conv, biases)
actived_conv = tf.nn.relu(bias)