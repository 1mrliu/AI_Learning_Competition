# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/7 上午11:38'

import pandas as pd
from sklearn import preprocessing

data = pd.read_csv('result.csv')
data.drop_duplicates(inplace=True)
data = pd.DataFrame(data)
lbl =  preprocessing.LabelEncoder()
# 将item_category_list 进行分割
# data_item_category = data['item_category_list'].str.split(';', expand=True).stack().reset_index().rename(columns={0:'item_category'})
# data.drop('item_category_list', axis=1, inplace=True)
for i in range(1, 3):
    data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category_list的第0列全部都一样
print(data.head(5))