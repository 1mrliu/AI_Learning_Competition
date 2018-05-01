# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/30 下午3:54'
import tensorflow as tf
'''
举例对比：
使用TensorFlow实现一个卷积层
使用TensorFlow-Slim实现同样结构的神经网络
'''
# 直接使用TensorFlow原始API实现卷积层
scope_name = None
with tf.variable_scope(scope_name):
    weights = tf.get_variable("weights", ...)
    biases = tf.get_variable("bias", ...)
    conv = tf.nn.conv2d(...)
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases))

# 使用TensorFlow-Slim实现卷积层 一行实现前向传播算法
# 首先加载slim模块 slim.conv2d函数中有三个参数
# 第一个为节点矩阵  第二个为当前卷积层过滤器的深度 第三个为过滤器的尺寸
# 可选的参数为步长 是否使用全0填充 激活函数的选择以及变量的命名空间
slim = tf.contrib.slim()
net = slim.conv2d(input, 32, [3, 3])
