"""
Create:LiuDong 
Update:2018-3-11
使用比XGB效率更高的lightGBM方法实现这个
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import train_test_split


# 1.加载数据
start_time = time.time()
dtypes ={
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
}

train_data = pd.read_csv('/Users/liudong/Desktop/talkingdata/train_sample.csv',dtype=dtypes)

test_data = pd.read_csv('/Users/liudong/Desktop/talkingdata/test.csv')
print('[{}] Finished  to load data!'.format(time.time()-start_time))

# 2.对数据中的日期进行处理
# 处理训练集的数据
train_data['day'] = pd.to_datetime(train_data['click_time']).dt.day.astype('uint8')
train_data['hour'] = pd.to_datetime(train_data['click_time']).dt.hour.astype('uint8')
train_data.drop(['click_time'], axis=1, inplace=True)

# 处理测试集的数据
print('Deal with the date ...')
test_data['day'] = pd.to_datetime(test_data['click_time']).dt.day.astype('int')
test_data['hour'] = pd.to_datetime(test_data['click_time']).dt.hour.astype('int')
test_data.drop(['click_time'], axis=1, inplace=True)


train_Y = train_data['is_attributed']
train_data.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)


sub = pd.DataFrame()
sub['click_id'] = test_data['click_id'].astype('uint32')
# test_data.drop('click_id', axis=1, inplace=True)

# 3.开始训练lightGBM
print('[{}] Start lightGBM Training'.format(time.time() - start_time))
params = {
    'num_leaves': 31,
    'objective': 'binary',
    'min_data_in_leaf': 200,
    'learning_rate': 0.1,
    'bagging_fraction': 0.85,
    'feature_fraction': 0.8,
    'bagging_freq': 3,
    'metric': 'auc',
    'num_threads': 4,}
MAX_ROUNDS = 600
x1, x2, y1,y2 = train_test_split(train_data, train_Y, test_size=0.1, random_state=99)

dtrain = lgb.Dataset(x1, label=y1)
dval = lgb.Dataset(x2, label=y2, reference=dtrain)
print('Start...Train')
model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],
                  early_stopping_rounds=1000, verbose_eval=10)
print('[{}] Finish LightGBM Training'.format(time.time() - start_time))

# 4.输出结果
sub['is_attributed'] = model.predict(test_data, num_iteration= model.best_iteration or  MAX_ROUNDS)
name = 'lightGBM'
sub.to_csv('result_%s.csv' % name, index=False)





