# -*- coding: utf-8 -*-
import tensorflow as tf
__author__ = 'liudong'
__date__ = '2018/4/26 下午3:31'
'''
mnist_inference 定义了前向传播过程以及神经网络中的参数
完整的tensorflow模型
可扩展性好  针对变量越来越多的情况来使用
持久化模型  模型可以持续使用
训练过程中可以隔一顿时间保存数据，防止意外死机
'''

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights',shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

# 定义神经网络的前向传播
def inference(input_tensor, regularizer):
    # 第一层神经网络 并完成前向传播
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # 第二层神经网络  并完成前向传播
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    return layer2
