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
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 定义模型的保存路径和文件名
MODEL_SAVE_PATH = "/Users/liudong/Desktop/model"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32,[None, mnist_inference.INPUT_NODE],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE],
                        name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用mnist_inference中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage (MOVING_AVERAGE_DECAY,
                                                           global_step)
    variable_average_op = variable_averages.apply (tf.trainable_variables ())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits (logits=y,
                                                                    labels=tf.argmax (y_, 1))
    cross_entropy_mean = tf.reduce_mean (cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay (
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    # loss 为交叉熵和L2正则化损失之和
    train_step = tf.train.GradientDescentOptimizer (learning_rate).minimize (loss,
                                                        global_step=global_step)
    # cross_entropy_mean 为交叉熵的值
    with tf.control_dependencies ([train_step, variable_average_op]):
        train_op = tf.no_op (name='train')
    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})
            # 每一千轮保存一次
            if i % 1000 == 0:
                print("After %d training step(s), loss on training"
                      "batch is %g" % (step, loss_value))
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step)
def main(argv=None):
    mnist = input_data.read_data_sets ('/path/to/MNIST_data', one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()
