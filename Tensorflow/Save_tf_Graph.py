# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/26 下午2:30'

import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    '''
    # 将模型保存到指定路径的文件夹
    sess.run(init_op)
    saver.save(sess, '/Users/liudong/Desktop/model/model.ckpt')
    '''

    # 加载已经保存的模型
    saver.restore(sess, '/Users/liudong/Desktop/model/model.ckpt')
    print(sess.run(result))




