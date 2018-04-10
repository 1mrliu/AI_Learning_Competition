'''
# 一个完整的-----训练神经网络解决二分类问题的
# 前向传播和反向传播都有
# @
'''
import tensorflow as tf
# 使用Numpy工具包生成模拟数据集
from numpy.random import RandomState

# 定义训练数据集的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法
# 使用sigmoid 函数将y转换为0~1之间的数值。 转换后y可以代表预测是正样本的概率，1-y代表预测是负样本的概率
# cross_entropy 定义了真实值和预测值之间的交叉熵
# 0.001是学习率
y = tf.sigmoid(y)
# clip_by_value 将值限制在 1e-10和1.0之间
# tf.log 对张量中的元素取对数
# * 代表元素之间直接相乘
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1-y)* tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState()
dataset_size = 128
X = rdm.rand(dataset_size, 2)
print(X)

Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行TensorFlow 程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    # 设定训练的轮数
    STEPS = 4000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
             # 每隔一段时间计算在所有数据上的  交叉熵
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print("After %d training step(s), cross entropy on all data is %g " %(i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))







