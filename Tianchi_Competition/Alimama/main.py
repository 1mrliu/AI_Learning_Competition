import time
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import log_loss
import warnings

# 设置warnings不可见
warnings.filterwarnings("ignore")


# 转换日期方法
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

# 对日期的值进行处理
def convert_data(data):
    # context_timestamp 表示广告商品的展示时间
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    # time  2018-09-18 10:09:04   day 8-9   hour 11-12
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))

    # user_query_day_hour
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(
        columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0:'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])
    return data

# 处理category_list，进行分割数据
def deal_category_list(data):
    lbl = preprocessing.LabelEncoder ()
    print('___________^item^____________')
    for i in range (1, 3):
        data['item_category_list' + str (i)] = lbl.fit_transform (data['item_category_list'].map (
            lambda x: str (str (x).split (';')[i]) if len (str(x).split (';')) > i else ''))  # item_category_list的第0列全部都一样
    for i in range (10):
        data['item_property_list' + str (i)] = lbl.fit_transform (data['item_property_list'].map (
            lambda x: str (str (x).split (';')[i]) if len (str (x).split (';')) > i else ''))
    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = lbl.fit_transform (data[col])

    print ('___________^context^____________')
    for i in range (5):
        data['predict_category_property' + str (i)] = lbl.fit_transform (data['predict_category_property'].map (
            lambda x: str (str (x).split (';')[i]) if len (str (x).split (';')) > i else ''))

    data['context_page0'] = data['context_page_id'].apply (
        lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4007  else 2)

    print ('___________^shop^____________')
    for col in ['shop_id']:
        data[col] = lbl.fit_transform (data[col])
    data['shop_score_delivery0'] = data['shop_score_delivery'].apply (lambda x: 0 if x <= 0.98 and x >= 0.96  else 1)
    return  data

if __name__ == "__main__":
    online = False
    # 这里用来标记是 线下验证 还是 在线提交
    data = pd.read_csv('/Users/liudong/Desktop/round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)
    data = deal_category_list(data)
    print('Start !!!')
    # print(data[:10])

    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    else:
        train = data.copy()
        test = pd.read_csv ('/Users/liudong/Desktop/round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = convert_data(test)
        test = deal_category_list(test)

    features = [# item
                'item_id', 'item_brand_id', 'item_city_id', 'item_price_level',
                'item_sales_level','item_collected_level', 'item_pv_level',
                # user
                'user_id','user_gender_id','user_age_level','user_occupation_id',
                'user_star_level', 'user_query_day','user_query_day_hour',
                'context_id', 'context_page_id', 'hour',
                # shop
                'shop_id','shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
                'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                # 处理  item_category_list
                'item_category_list1','item_category_list2',
                'item_property_list0','item_property_list1','item_property_list2','item_property_list3',
                'item_property_list4','item_property_list5','item_property_list6','item_property_list7',
                'item_property_list8','item_property_list9',
                # 处理  context_category_list
                'predict_category_property0', 'predict_category_property1','predict_category_property2', 'predict_category_property3', 'predict_category_property4',
                # context_page
                'context_page0',
                ]
    target = ['is_trade']

    if online == False:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit (train[features], train[target], feature_name=features, categorical_feature=['user_gender_id', ])
        test['lgb_predict'] = clf.predict_proba(test[features], )[:, 1]
        print('出现的误差是：', log_loss(test[target], test['lgb_predict']))
    else:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[features], train[target], feature_name=features, categorical_feature=['user_gender_id', ])
        test['predicted_score'] = clf.predict_proba(test[features])[:, 1]
        test[['instance_id', 'predicted_score']].to_csv('baseline.csv', index=False, sep=' ')

