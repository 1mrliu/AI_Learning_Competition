# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/2 下午4:01'
import  numpy as np

# 定义RNN的参数
X = [1, 2]
state = [0.0, 0.0]
# 分开不同输入部分的权重以方便操作
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([[0.5, 0.5]])
b_cell = np.asarray([0.1, -0.1])

# 定义用于输出的全连接层参数
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 按照时间顺序执行RNN的前向传播过程
for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)

    # 根据当前状态计算最终输出
    final_output = np.dot(state, w_output) + b_output

    # 输出每个时刻的信息
    print("before activation:", before_activation)
    print("state:", state)
    print("output:", final_output)
