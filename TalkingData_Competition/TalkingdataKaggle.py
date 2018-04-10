# -*- coding: utf-8 -*-
"""
Create:LiuDong 2018-3-8
Update:LiuDong 2018-3-10
使用XGB算法实现
"""
import time
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import xgboost as xgb


# 定义处理时间的函数
def dataPreProcessTime(data):
    data['click_time'] = pd.to_datetime(data['click_time']).dt.date
    data['click_time'] = data['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    return data

# 1.加载数据
start_time = time.time()

train_data = pd.read_csv('/Users/liudong/Desktop/talkingdata/train_sample.csv')

test_data = pd.read_csv('/Users/liudong/Desktop/talkingdata/test.csv')

print('[{}] Finished  to load data!'.format(time.time()-start_time))


# 2.对数据中的日期进行处理
train_data = dataPreProcessTime(train_data)
test_data  = dataPreProcessTime(test_data)
print(train_data.head(10))

train_Y = train_data[['is_attributed']].as_matrix()
train_data.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

train_X = train_data[['ip', 'app', 'device', 'os', 'channel']].as_matrix()


# svc = SVC(C=1.0, kernel='rbf', decision_function_shape='ovo')

# svc = svc.fit(train_X,train_Y)



test_X = test_data[['ip','app', 'device', 'os', 'channel']].as_matrix()

# predictions = svc.predict(test_X).astype(np.double)

sub = pd.DataFrame({'click_id':test_data['click_id'].as_matrix()})
test_data.drop('click_id', axis=1, inplace=True)

# 3.开始训练XGBoost
print('[{}] Start XGBoost Training'.format(time.time() - start_time))
params = {'eta': 0.1,
          'max_depth': 4,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':4,
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'random_state': 99,
          'silent': True}

x1, x2, y1,y2 = train_test_split(train_data, train_Y, test_size=0.1, random_state=99)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 270, watchlist, maximize=True, verbose_eval=10)

print('[{}] Finish XGBoost Training'.format(time.time() - start_time))

# 4.输出结果
sub['is_attributed'] = model.predict(xgb.DMatrix(test_data))
name = 'xgb_sub'
sub.to_csv('result_%s.csv' % name, index=False)





