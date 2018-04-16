# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/15 下午9:13'

import tensorflow as tf
# 定义了一个实数变量用于计算滑动平均，这个变量的初始值是0.
# 手动制定了变量的类型为tf.float32 因为所有需要计算滑动平均的变量必须是实数型
v1 = tf.Variable(0, dtype=tf.float32)
# 这里step变量的模拟神经网络迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类（class） 初始化时给定了衰减率（0.99）和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作。这里需要设置一个列表，每次执行这个操作时
# 这个列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])
with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过ema.average(v1)获取滑动平均之后变量的取值 在初始化之后变量v1的值和v1的滑动平均都为0
    print(sess.run([v1, ema.average(v1)])) # [0.0, 0.0]

    # 更新变量v1的值到5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值 衰减率为min{0.99, (1+step)/(10+step)=0.1}=0.1
    # 所以v1的滑动平均会被更新为0.1*0+0.9*5 =4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)])) # [5.0, 4.5]

    # 更新step的值为10000
    sess.run(tf.assign(step, 10000))

    # 更新变量v1的值到10
    sess.run (tf.assign (v1, 10))
    # 更新v1的滑动平均值 衰减率为min{0.99, (1+step)/(10+step)=0.99}=0.99
    # 所以v1的滑动平均会被更新为0.99*4.5+0.01*10 =4.555
    sess.run (maintain_averages_op)
    print(sess.run ([v1, ema.average (v1)]))  # [10.0, 4.5549998]

    # 再次更新滑动平均值，得到的新滑动平均值为0.99*4.555+0.01*10=4.60945
    sess.run (maintain_averages_op)
    print(sess.run ([v1, ema.average (v1)]))  # [10.0, 4.6094499]


