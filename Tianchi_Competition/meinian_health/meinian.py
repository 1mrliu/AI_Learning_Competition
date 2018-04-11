# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/11 下午5:04'

import pandas as pd
import lightgbm as lgb

# 读取数据 处理数据
data1  = pd.read_csv('/Users/liudong/Desktop/meinian/meinian_round1_data_part1_20180408.txt', sep='$')
data2  = pd.read_csv('/Users/liudong/Desktop/meinian/meinian_round1_data_part2_20180408.txt',sep='$')
data_train = pd.read_csv('/Users/liudong/Desktop/meinian/meinian_round1_train_20180408.csv',encoding="gbk")
data_test  = pd.read_csv('/Users/liudong/Desktop/meinian/meinian_round1_test_a_20180409.csv',encoding="gbk")
data = pd.concat([data1,data2], ignore_index=True,)
data = pd.DataFrame(data)
# 去重
data = data.drop_duplicates(['vid'])
data = data.drop_duplicates(['table_id'])
features = ['table_id']
train_X = data[features]
train_Y = data_train['收缩压']
print(train_X)
# print(train_Y)
# test = data_train['vid']
print(data_train.head(5))
print(data.describe())
clf = lgb.LGBMClassifier(num_leaves=100, max_depth=9, n_estimators=90, n_jobs=20)
clf.fit(train_X, train_Y)
data_test['收缩压'] = clf.predict_proba(data_test['vid'])
data_test[['vid', '收缩压']].to_csv('baseline.csv', index=False, sep=' ')