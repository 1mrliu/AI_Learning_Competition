import time
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestRegressor
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
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0:'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])

    return data


# 处理category_list，进行分割数据
def deal_category_list(data):
    lbl = preprocessing.LabelEncoder ()

    print('>>>>>>>>>>>>>>>item>>>>>>>>>>>>>>>')
    for i in range(1, 3):
        data['item_category_list' + str (i)] = lbl.fit_transform (data['item_category_list'].map (
            lambda x: str (str (x).split (';')[i]) if len (str(x).split (';')) > i else ''))  # item_category_list的第0列全部都一样
    del data['item_category_list']

    for i in range(10):
        data['item_property_list' + str (i)] = lbl.fit_transform (data['item_property_list'].map (
            lambda x: str (str (x).split (';')[i]) if len (str (x).split (';')) > i else ''))
    del data['item_property_list']
    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = lbl.fit_transform (data[col])

    print('>>>>>>>>>user>>>>>>>>>>>')
    data['user_gender'] = data['user_gender_id'].apply(lambda x: 1 if x==-1 else 2)
    data['user_age'] = data['user_age_level'].apply(lambda x: x-1000 if x>=1000 else 0)
    data['user_star'] = data['user_star_level'].apply(lambda x: x-3000 if x >=3000 else 0)

    print ('>>>>>>>>>>>>context>>>>>>>>>>>>>')
    for i in range(5):
        data['predict_category_property' + str (i)] = lbl.fit_transform (data['predict_category_property'].map (
            lambda x: str (str (x).split (';')[i]) if len (str (x).split (';')) > i else ''))
    data['context_page0'] = data['context_page_id'].apply (
        lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4007  else 2)

    print ('>>>>>>>>>>>>>shop>>>>>>>>>>>>>')
    for col in ['shop_id']:
        data[col] = lbl.fit_transform(data[col])
    data['shop_score_delivery'] = data['shop_score_delivery'].apply (lambda x: 0 if x <= 0.98 and x >= 0.96  else 1)
    data['shop_review_num'] = data['shop_review_num_level'].apply(lambda x: x if x<10 else x-10)
    data['shop_star_level'] = data['shop_star_level'].apply(lambda x: x-5000)
    print(data[:5])
    return data


# 缺失值处理函数
def set_missing_feature(train_for_missingkey, data, info):
    known_feature = train_for_missingkey[train_for_missingkey.Age.notnull()].as_matrix()
    unknown_feature = train_for_missingkey[train_for_missingkey.Age.isnull()].as_matrix()
    y = known_feature[:, 0] # 第1列作为待补全属性
    x = known_feature[:, 1:] # 第2列及之后的属性作为预测属性
    # 随机森林回归  第一个参数是随机数  第二个参数是在森林中树的个数
    rf = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    # 从训练集建立森林
    rf.fit(x, y)
    print(info, "缺失值预测得分", rf.score(x, y))
    predictage = rf.predict(unknown_feature[:, 1:])
    data.loc[data.Age.isnull(), 'Age'] = predictage
    return data


# 组合特征
def merge_structure(data):
    # 代表了商品的价格和销量的相关性
    data['item_price_sales'] = list(map(lambda x,y: merge_corr(x,y), data['item_price_level'], data['item_sales_level'] ))
    # 转换为哑变量的形式
    # data['item_price_sales'] = pd.get_dummies(data['item_price_sales'])

    # 年龄和商品价格等级
    data['price_age'] = list (map (lambda x, y: merge_corr(x, y), data['item_price_level'], data['user_age_level']))
    # 转换为哑变量的形式
    # data['price_age'] = pd.get_dummies (data['price_age'])

    # 商品价格等级和用户职业
    data['price_occupation'] = list (map (lambda x, y: merge_corr(x, y), data['item_price_level'], data['user_occupation_id']))
    # 哑变量
    # data['price_occupation'] = pd.get_dummies (data['price_occupation'])

    # 广告商品的展示时间和商品被收藏的
    data['context_timestamp_collected'] = list (map (lambda x, y: merge_corr (x, y), data['hour'], data['item_collected_level']))

    # 用户性别和广告商品的品牌
    data['user_sex_brand'] = list (map (lambda x, y: merge_corr (x, y), data['user_gender_id'], data['item_brand_id']))


    return data



# 计算组合特征之间的关系
def merge_corr(x,y):
    # 计算相关系数
    result = (pow(abs(x)-abs(y),2))/2
    return result


if __name__ == "__main__":
    online = False
    # 这里用来标记是 线下验证 还是 在线提交
    data = pd.read_csv('/Users/liudong/Desktop/round1_ijcai_18_train_20180301.txt', sep=' ')

    data.drop_duplicates(inplace=True)

    data = convert_data(data)

    data = deal_category_list(data)
    data = merge_structure(data)

    if online == False:
        # train, test = train_test_split(data, test_size= 0.3, random_state=1)
        train = data.loc[data.day < 24]
        test = data.loc[data.day == 24]
    else:
        train = data.copy()
        test = pd.read_csv ('/Users/liudong/Desktop/round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = convert_data(test)
        test = deal_category_list(test)

    features = [  # item
            'item_id', 'item_city_id', 'item_price_level',
            'item_sales_level', 'item_collected_level', 'item_pv_level',
            # user
            'user_id', 'user_gender', 'user_age', 'user_occupation_id',
            'user_star',
            # shop
            'shop_id', 'shop_review_num', 'shop_review_positive_rate', 'shop_star_level',
            # context
            'context_id','context_page_id','predict_category_property4',
            # 组合特征
            'item_price_sales','price_age','price_occupation','context_timestamp_collected','user_sex_brand',
    ]

    target = ['is_trade']

    if online == True:

        print ('--------Start LGB_Model-------')
        # num_leaves=70, max_depth=8, n_estimators=90, n_jobs=20
        clf = lgb.LGBMClassifier(num_leaves=100, max_depth=9, n_estimators=90, n_jobs=20)
        clf.fit (train[features], train[target], feature_name=features, categorical_feature=['user_gender', ])
        test['lgb_predict'] = clf.predict_proba(test[features], )[:, 1]
        print('LGB出现的误差是：', log_loss(test[target], test['lgb_predict']))
        # 0.0883036  0.082581 0.082731 0.082712 0.0826939 0.08265261  0.082625684 0.08251228993595107 0.08250761885627189
        print ('--------Start CatBoost_Model-------')
        # num_leaves=70, max_depth=8, n_estimators=90, n_jobs=20
        clf = lgb.LGBMClassifier (num_leaves=100, max_depth=8, n_estimators=90, n_jobs=30)
        clf.fit (train[features], train[target], feature_name=features, categorical_feature=['user_gender', ])
        test['lgb_predict'] = clf.predict_proba (test[features], )[:, 1]
        print ('LGB出现的误差是：', log_loss (test[target], test['lgb_predict']))



    else:

        clf = lgb.LGBMClassifier(num_leaves=100, max_depth=9, n_estimators=90, n_jobs=20)
        clf.fit(train[features], train[target], feature_name=features, categorical_feature=['user_gender', ])
        test['predicted_score'] = clf.predict_proba(test[features])[:, 1]
        test[['instance_id', 'predicted_score']].to_csv('baseline.csv', index=False, sep=' ')