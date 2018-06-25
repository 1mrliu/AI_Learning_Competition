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
LAYER2_NODE = 500 # 定义第二层

LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001 # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000 # 训练的轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率


def inference(input_tensor, reuse=False):
    # 定义第一层神经网络的变量  输入层到第一层神经网络的  权重和偏置项  relu激活函数实现去线性化
    with tf.variable_scope('layer1', reuse=reuse):
        # get_variable(name,shape,initializer)
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        # 对计算的值用激活函数实现去线性化操作
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
    # 定义第二层神经网络的变量和前向传播算法
    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, LAYER2_NODE],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER2_NODE],
                                      initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    # 定义第三层神经网络
    with tf.variable_scope('layer3', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER2_NODE,OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer3 = tf.nn.relu(tf.matmul(layer2,weights) + biases)
    return layer3



def train(mnist):
    # 定义placeholder作为存放输入数据的地方  INPUT_NODE为784  OUTPUT_NODE为10
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    # 生成隐藏
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))

    y = inference(x)
    # 训练神经网络的时候，一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable( 0, trainable=False)
    # 给滑动平均衰减率和训练论述的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均 trainanle_variables代表的就是集合
    # 集合中的元素就是没有指定trainable为false的参数
    variable_average_op = variable_averages.apply(tf.trainable_variables())
    #
    average_y = inference(x, True)
    # 交叉熵被用来刻画预测值和真实值之间的差距的损失函数
    # 第一个参数是不包含softmax层的前向传播结果y  第二个y_是训练数据的正确答案
    #
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在batch中所有交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # L2正则化防止过拟合
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失率等于正则化损失和交叉熵损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, # 基础的学习率，随着迭代的进行更新变量时使用的学习率在这个基础上递减
        global_step,# 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,# 过完所有的数据集需要迭代的次数
        LEARNING_RATE_DECAY # 学习率衰减速度
    )
    # GDO来优化损失函数   loss 为交叉熵和L2正则化损失之和
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # cross_entropy_mean 为交叉熵的值
    # train_step = tf.train.GradientDescentOptimizer (learning_rate).minimize
    # (cross_entropy_mean, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')
    # averagy代表一个batchsize*10的二维数组，每一行代表一个样例的前向传播结果
    # tf.argmax的第二个参数代表选取最大值的操作只在第一个维度中进行，每一行选取最大值对应的下标
    # equal返回bool值
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 将布尔值转化为实数型，然后计算出平均值(模型在这一数组上的正确率)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}
        # 迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            # 每一千轮输出一次在验证集上的测试结果
            if i % 1000 ==0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy'
                      'using average model is %g' %(i, validate_acc))
            # 产生这一轮使用的一个batch的训练数据，并进行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test accuracy using average'
          'model is %g '%(TRAINING_STEPS, test_acc))

        validate_acc = sess.run(accuracy, feed_dict=validate_feed)
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, validation accuracy using average"
          "model is %g, test accuracy using average model is %g"%(i, validate_acc, test_acc))

def main(argv=None):
    train(mnist)

if __name__ == '__main__':
    tf.app.run()




