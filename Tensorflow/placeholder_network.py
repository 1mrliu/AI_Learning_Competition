'''
使用placeholder实现前向传播算法

'''
import tensorflow as tf

# 声明w1、w2两个变量

w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 定义placeholder 作为存放输入数据的地方 这里的维度不一定要定义
# 如果维度是确定的，那么给出维度可以降低出错的概率
x = tf.placeholder(tf.float32, shape=(3, 2), name='input')

# 通过前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


sess = tf.Session()

# 首先对w1, w2进行初始化
init_op = tf.global_variables_initializer()
sess.run(init_op)

# 输出结果

print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
sess.close()
