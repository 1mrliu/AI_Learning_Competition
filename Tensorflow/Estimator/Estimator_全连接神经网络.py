# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/5 上午11:06'
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 将TensorFlow的日志信息输出到屏幕
tf.logging.set_verbosity(tf.logging.INFO)
mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=False)

# 指定神经网络的输入层 这所有的输入都会拼接在一起作为整个神经网络的输入
feature_columns = [tf.feature_column.numeric_column("image", shape=[784])]
# 定义Estimator模型
estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[500],
    n_classes=10,
    optimizer=tf.train.AdamOptimizer(),
    model_dir="../Keras/log")
# 定义数据输入
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=None,
    batch_size=128,
    shuffle=True)
# 训练模型
estimator.train(input_fn=train_input_fn, steps=10000)
# 定义测试时的数据输入
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": mnist.test.images},
    y=mnist.test.labels.astype(np.int32),
    num_epochs=1,
    batch_size=128,
    shuffle=False)

# 通过evaluate评测训练好的模型的效果
accuarcy_score = estimator.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest accuracy: %g %%" % (accuarcy_score*100))


