# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/1 上午10:48'
import matplotlib.pyplot as plt
import tensorflow as tf
'''
TensorFLow中对图像的处理函数
'''

# 需要把原文中的'r' 改为 'rb' 来使用，才可以达到预期的效果
image_raw_data = tf.gfile.FastGFile("../CNN/cat.jpg", 'rb').read()
with tf.Session() as sess:
    image_data = tf.image.decode_jpeg(image_raw_data)
    print(image_data.eval())

    # 使用pyplot工具可视化得到的图像
    plt.imshow(image_data.eval())
    plt.show()

    # 将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中 可以得到和原始图像一样的图像
    encoded_image = tf.image.encode_jpeg(image_data)
    with tf.gfile.GFile("../CNN/cat.jpeg", "wb") as f:
        f.write(encoded_image.eval())
    # 对原始图像数据进行初始化，数据处理为规范统一的类型
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    resized = tf.image.resize_images(image_data, [300, 300], method=3)
    # method  0:双线性插值法 1：最近邻居法 2：双三次插值法 3:面积插值法
    plt.imshow(resized.eval ())
    plt.show()

    # 对图像进行裁剪或者填充
    croped = tf.image.resize_image_with_crop_or_pad(image_data, 1000, 1000)
    padded = tf.image.resize_image_with_crop_or_pad (image_data, 3000, 3000)
    plt.imshow(croped.eval())
    plt.show()
    plt.imshow(padded.eval())
    plt.show()