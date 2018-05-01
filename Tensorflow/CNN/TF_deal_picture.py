# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/1 上午10:48'
import matplotlib.pyplot as plt
import tensorflow as tf

# 需要把原文中的'r' 改为 'rb' 来使用，才可以达到预期的效果
image_raw_data = tf.gfile.FastGFile("/Users/liudong/Desktop/picture/cat.jpg", 'rb').read()
with tf.Session() as sess:
    image_data = tf.image.decode_jpeg(image_raw_data)
    print(image_data.eval())

    # 使用pyplot工具可视化得到的图像
    plt.imshow(image_data.eval())
    plt.show()

    # 将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中 可以得到和原始图像一样的图像
    encoded_image = tf.image.encode_jpeg(image_data)
    with tf.gfile.GFile("/Users/liudong/Desktop/picture/cat.jpeg", "wb") as f:
        f.write(encoded_image.eval())