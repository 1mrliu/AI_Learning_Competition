# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/5 下午1:56'
import tensorflow as tf


# 定义一个简单的计算图，实现向量加法的操作
input1 = tf.constant([1.0, 2.0, 3.0], name="input1")

input2 = tf.Variable(tf.random_uniform([3]), name="input2")

output = tf.add_n([input1, input2], name="add")

session =tf.Session()
writer = tf.summary.FileWriter('../Tensorboard_可视化/log', tf.get_default_graph())
init = tf.global_variables_initializer()
writer.close()
session.run(init)


