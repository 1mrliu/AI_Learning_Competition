# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/15 7:46 PM'

import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

train_data = pd.read_csv("/Users/liudong/Desktop/Zhengqi_train.txt",sep="\t")
test_data = pd.read_csv("/Users/liudong/Desktop/Zhengqi_test.txt",sep="\t")
# print(train_data.head())
# print(test_data.head())
train_Y = train_data['target']
train_X = train_data.drop(['target'],axis=1)

# train_X, val_X, train_Y, val_Y = train_test_split(train_data,Y,test_size=0.3,random_state=42)
# print(train_X.head())
clf  = lgb.LGBMClassifier()
clf.fit(train_X,train_Y.astype(int))
predict_result = clf.predict(test_data)
result  = pd.DataFrame({'target':predict_result})
result.to_csv('./result.txt',index=False)