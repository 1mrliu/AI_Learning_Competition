# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载前向传播的函数和常量
import mnist_inference
__author__ = 'liudong'
__date__ = '2018/4/26 下午3:33'
'''
mnist_train 定义了神经网络的训练过程
完整的tensorflow模型
可扩展性好  针对变量越来越多的情况来使用
持久化模型  模型可以持续使用
训练过程中可以隔一顿时间保存数据，防止意外死机
'''
BATCH_SIZE = 100
