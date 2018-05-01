# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/1 下午5:24'
import tensorflow as tf
import numpy as np
import threading
import time
# 队列进行处理数据
q = tf.FIFOQueue(2, "int32")
init = q.enqueue_many(([0, 10],))
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])
with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(v)

# 线程中运行的程序，这个程序每隔一秒判断是否需要停止并打印自己的ID

def MyLoop(coord, worker_id):
    # 使用tf.Coordinator类提供的协同工具判断当前线程是否需要停止
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stopping from id: %d\n" % worker_id)
            # 调用request_stop()函数来通知其他线程停止
            coord.request_stop()
        else:
            print("Working on id: %d\n" % worker_id)
        # 暂停一秒
        time.sleep(1)
coord = tf.train.Coordinator()
threads = [ threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5)]
for t in threads:
    t.start()
coord.join(threads)
