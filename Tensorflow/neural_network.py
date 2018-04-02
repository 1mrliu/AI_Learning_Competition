'''
# 通过变量实现神经网络的参数并实现前向传播的过程
@auther liudong
@ 2018-4-2
'''
import tensorflow as tf

# 声明w1、w2两个变量  通过设置seed参数设定了随机种子
# 这样可以保证每次运算得到的结果是一样的
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量   x为一个1*2的矩阵
x = tf.constant([[0.7, 0.9]])

# 通过前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


sess = tf.Session()

# 首先对w1, w2进行初始化
sess.run(w1.initializer)
sess.run(w2.initializer)

# 输出结果
print(sess.run(y))
sess.close()
