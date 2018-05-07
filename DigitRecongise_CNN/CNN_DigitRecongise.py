# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/7 下午3:47'
'''
CNN实现数字识别
Kaggle
'''
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
# 将那些用matplotlib绘制的图显示在页面里而不是弹出一个窗口
# %matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # 转换成 one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Load the data
train = pd.read_csv('/Users/liudong/Downloads/train.csv')
test = pd.read_csv('/Users/liudong/Downloads//test.csv')

X_train = train.values[:,1:]
Y_train = train.values[:,0]
test = test.values
# print(X_train.dtype)
# print(test.dtype)
# Normalization
X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
test = test.reshape(-1, 28, 28, 1)
# one-hot encode
Y_train = to_categorical(Y_train, num_classes=10)
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)
# CNN model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 30
batch_size = 86
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
# 设置图像翻转
datagen = ImageDataGenerator(
        featurewise_center=False,  # 使输入数据集去中心化（均值为0）, 按feature执行
        samplewise_center=False,  # 使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # 将输入除以数据集的标准差以完成标准化, 按feature执行
        samplewise_std_normalization=False,  # 将输入的每个样本除以其自身的标准差
        zca_whitening=False,  # 对输入数据施加ZCA白化
        rotation_range=10,  # 数据增强时图片随机转动的角度
        zoom_range = 0.1, # 随机缩放的幅度
        width_shift_range=0.1,  # 图片宽度的某个比例，数据增强时图片水平偏移的幅度
        height_shift_range=0.1,  # 图片高度的某个比例，数据增强时图片竖直偏移的幅度
        horizontal_flip=False,  # 进行随机水平翻转
        vertical_flip=False)  # 进行随机竖直翻转
datagen.fit(X_train)

starttime = datetime.datetime.now()

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

endtime = datetime.datetime.now()

print ((endtime - starttime).seconds)

results = model.predict(test)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("/Users/liudong/Downloads/Result_keras_CNN.csv",index=False)