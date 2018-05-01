# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/16 上午9:26'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
# print("Training data size:", mnist.train.num_examples)
# print("Validating data size:", mnist.validation.num_examples)
# print("Testing data size:", mnist.test.num_examples)
# print("Example training data:", mnist.train.images[0])
# print("Example training data label:", mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print('X shape:', xs.shape)
print('Y shape:', ys.shape)

# MNIST数据集想关常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER1_NODE = 500 # 隐藏层节点数 这里只使用一个隐藏层 隐藏层有500个节点
BATCH_SIZE = 100 # 一个训练batch中的训练数据个数 数字越小时，训练过程越接近随机梯度下降
                 # 数字越大时，训练数据越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


# def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
#
#     if avg_class == None:
#         layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
#         return tf.matmul(layer1, weights2) + biases2
#     else:
#         # relu 函数实现去线性化
#         layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+
#                             avg_class.average(biases1))
#         return tf.matmul(layer1,avg_class.average(weights2))+ avg_class.average(biases2)
def inference(input_tensor, reuse=False):
    # 定义第一层神经网络的变量和前向传播算法
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
    # 定义第二层神经网络的变量和前向传播算法
    with tf.variable_scope ('layer2', reuse=reuse):
        weights = tf.get_variable ("weights", [LAYER1_NODE, OUTPUT_NODE],
                                       initializer=tf.truncated_normal_initializer (stddev=0.1))
        biases = tf.get_variable ("biases", [OUTPUT_NODE],
                                      initializer=tf.constant_initializer (0.0))
        layer2 = tf.nn.relu (tf.matmul(layer1, weights) + biases)
    return layer2

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    # 生成隐藏层的个数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    # y = inference(x, None, weights1, biases1, weights2, biases2)
    y = inference(x)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_averages.apply(tf.trainable_variables())
    # average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    average_y = inference(x, True)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    # loss 为交叉熵和L2正则化损失之和
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # cross_entropy_mean 为交叉熵的值
    # train_step = tf.train.GradientDescentOptimizer (learning_rate).minimize
    # (cross_entropy_mean, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 ==0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy'
                      'using average model is %g' %(i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

    test_acc = sess.run(accuracy, feed_dict=test_feed)
    print('After %d training step(s), test accuracy using average'
          'model is %g '%(TRAINING_STEPS, test_acc))

    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
    test_acc = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training steps, validation accuracy using average"
          "model is %g, test accuracy using average model is %g"
          % (i, validate_acc, test_acc))

def main(argv=None):
    train(mnist)

if __name__ == '__main__':
    tf.app.run()




