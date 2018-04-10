# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/7 上午11:38'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


data = pd.read_csv('result.csv')
data.drop_duplicates(inplace=True)
data = pd.DataFrame(data)
lbl =  preprocessing.LabelEncoder()

for i in range(1, 3):
    data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category_list的第0列全部都一样
def deal_user_star_level(data):
    data['user_start_level'] = data['user_star_level'].apply(lambda x: int(x-3000))
    return data
deal_user_star_level(data)

print(data.head(5))
data.drop(['item_category_list'], axis=1, inplace=True)
data.drop(['item_property_list'], axis=1, inplace=True)
data.drop(['predict_category_property'], axis=1, inplace=True)
features = [ # item
            'item_id', 'item_city_id', 'item_price_level',
            'item_sales_level','item_collected_level', 'item_pv_level',
            # user
            'user_id','user_gender_id','user_age_level','user_occupation_id',
            'user_star_level',
            # shop
            'shop_id','shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
           ]
data = data[features]
data_corr = data.corr(min_periods=10)

# 画出特征相关性的热力图
sns.set()
a = plt.figure('特征相关性热力图')
ax = plt.subplot(111)
heat_map = sns.heatmap(data_corr, vmin=-1, vmax=1, annot=True, square=True)
plt.plot(50,50)
plt.show()
plt.close()