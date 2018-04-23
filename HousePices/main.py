# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/23 下午2:28'

import pandas as pd
import lightgbm as lgb

# 加载数据
df_train = pd.read_csv('/Users/liudong/Desktop/house_price/train.csv')
df_test  = pd.read_csv('/Users/liudong/Desktop/house_price/test.csv')
print(df_train.columns)

# 对数据中的特征进行处理
features = [ 'MSSubClass',  'LotFrontage', 'LotArea',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'MasVnrArea','BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea','MiscVal', 'MoSold', 'YrSold'
        ]


# features = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

train_X = df_train[features]
train_Y = df_train['SalePrice']
print('##########开始训练数据##########')
lg = lgb.LGBMClassifier()
lg.fit(train_X, train_Y)

# 保存结果
result = pd.DataFrame()
result['Id'] = df_test['Id']
result['SalePrice'] = lg.predict(df_test[features])
# index=False 是用来除去行编号
result.to_csv('/Users/liudong/Desktop/house_price/result.csv', index=False)

print('##########结束训练##########')