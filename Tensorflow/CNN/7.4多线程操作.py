# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/2 下午2:38'
import tensorflow as tf

# 声明一个先进先出的对列
queue = tf.FIFOQueue(100, "float")

# 定义对列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

tf.train.add_queue_runner(qr)
# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用Coordinator来协同启动的线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(3):
        print(sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)
