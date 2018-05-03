# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/3 下午3:06'
import tensorflow as tf

# 定义一个基本的LSTM结构作为循环体的基础结构
# 深层循环神经网络可以使用不同的循环体结构
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
lstm_size = 10
number_of_layers = 5
batch_size=12
num_steps = 5
# 通过 MultiRNNCell 类实现深层循环网络中每一个时刻的前向传播过程
# number_of_layers 表示有多少层
# 初始化 MultiRNNCell 否则TensorFlow会在每一层之间共享数据
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
    [lstm_cell(lstm_size) for _ in range(number_of_layers)])
# 和经典的循环神经网络一样，可以通过zero_state函数来获取初始状态
state = stacked_lstm.zero_state(batch_size, tf.float32)

# 计算每一时刻的前向传播结果
for i in range(len(num_steps)):
    if i > 0:
        tf.get_variable_scope().reuse_variables()
    stacked_lstm_output, state = stacked_lstm(current_input, state)
    final_output = fully_connected(stacked_lstm_output)
    loss += calc_loss(final_output, expected_output)

