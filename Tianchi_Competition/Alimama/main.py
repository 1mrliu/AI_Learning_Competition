import time
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestRegressor
import gc
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
        lambda x: x-4000 if x>4000  else 2)

    print ('>>>>>>>>>>>>>shop>>>>>>>>>>>>>')
    for col in ['shop_id']:
        data[col] = lbl.fit_transform(data[col])
    data['shop_score_delivery'] = data['shop_score_delivery'].apply (lambda x: 0 if x <= 0.98 and x >= 0.96  else 1)
    data['shop_review_num'] = data['shop_review_num_level'].apply(lambda x: x if x<10 else x-10)
    data['shop_star_level'] = data['shop_star_level'].apply(lambda x: x-5000)
    # print(data[:5])
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


if __name__ == "__main__":
    online = False
    # False 线下验证    True线上测评
    data = pd.read_csv('/Users/liudong/Desktop/round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)
    data = deal_category_list(data)

    print("####################开始训练集中的特征融合#########################")

    # 商品品牌 价格 用户职业 销量
    print ('Merge 商品价格 品牌 用户职业 销量   structure!!!')
    gp = data[['item_price_level', 'item_brand_id', 'user_occupation_id','item_sales_level']].groupby (
        by=['item_price_level', 'item_brand_id', 'user_occupation_id'])[
        ['item_sales_level']].std().reset_index ().rename (index=str,
                                                            columns={'item_sales_level': 'price_occupation'})
    data = data.merge (gp, on=['item_price_level', 'item_brand_id', 'user_occupation_id'], how='left')
    del gp
    gc.collect ()
    # 哑变量
    # data['price_occupation'] = pd.get_dummies (data['price_occupation'])

    # 广告商品的展示时间  商品收藏 商品销量
    print ('Merge  广告商品的展示时间 收藏 销量 structure!!!')
    gp = data[['hour', 'item_collected_level', 'item_sales_level']].groupby (
        by=['hour', 'item_collected_level'])[
        ['item_sales_level']].std().reset_index ().rename (index=str,
                                                         columns={'item_sales_level': 'context_timestamp_collected'})
    data = data.merge (gp, on=['hour', 'item_collected_level'], how='left')
    del gp
    gc.collect ()
    # list (map (lambda x, y: merge_corr (x, y), data['hour'], data['item_collected_level']))

    # 用户性别和广告商品的品牌
    print ('Merge  用户性别、职业、广告商品的品牌 structure!!!')
    gp = data[['user_gender_id', 'user_occupation_id', 'item_brand_id']].groupby (by=['user_gender_id', 'user_occupation_id'])[
        ['item_brand_id']].mean().reset_index ().rename (index=str,
                                                            columns={'item_brand_id': 'user_sex_brand'})
    data = data.merge (gp, on=['user_gender_id', 'user_occupation_id'], how='left')
    del gp
    gc.collect ()

    # 商品品牌编号—城市编号——销量  var() 方差
    print('Merge item structure!!!')
    gp = data[['item_brand_id', 'item_city_id', 'item_sales_level']].groupby (by=['item_brand_id', 'item_city_id'])[
        ['item_sales_level']].var().reset_index ().rename (index=str, columns={'item_sales_level': 'item_brand_city_sales'})

    data = data.merge (gp, on=['item_brand_id', 'item_city_id'], how='left')
    del gp
    gc.collect()

    # 用户性别—年龄—职业-星级 mean()
    print ('Merge user structure!!!')
    gp = data[['user_gender_id', 'user_age_level', 'user_occupation_id','user_star_level']].groupby (by=['user_gender_id', 'user_age_level', 'user_occupation_id'])[
        ['user_star_level']].mean().reset_index ().rename (index=str, columns={'user_star_level': 'user_gender_age_occupation'})
    data = data.merge (gp, on=['user_gender_id', 'user_age_level', 'user_occupation_id'], how='left')
    data['user_gender_age_occupation'] = data['user_gender_age_occupation'].apply((lambda x: x-3000))
    del gp
    gc.collect ()


    # 店铺—服务态度—物流-描述-好评率 mean()
    print('Merge shop structure!!!')
    gp = data[['shop_score_service', 'shop_score_delivery', 'shop_score_description', 'shop_review_positive_rate']].groupby (
        by=['shop_score_service', 'shop_score_delivery', 'shop_score_description'])[
        ['shop_review_positive_rate']].std().reset_index ().rename (index=str,
                                                            columns={'shop_review_positive_rate': 'shop_service_delivery_review'})
    data = data.merge (gp, on=['shop_score_service', 'shop_score_delivery', 'shop_score_description'], how='left')
    del gp
    gc.collect ()

    # 上下文 展示时间和出现在展示页编号的可能性 mean()
    print ('Merge context structure!!!')

    gp = \
    data[['context_id', 'day','item_pv_level', 'hour', 'context_page0']].groupby (
        by=['context_id', 'day', 'item_pv_level', 'hour'])[
        ['context_page0']].mean().reset_index ().rename (index=str, columns={'context_page0': 'context_day_hour_page'})
    data = data.merge (gp, on=['context_id', 'day', 'item_pv_level','hour'], how='left')
    del gp
    gc.collect ()


    print ("####################结束训练数据集特征融合#########################")
    print(data[:10])

    if online == False:
        # 线下预测
        train, test = train_test_split(data, test_size= 0.3, random_state=1)
        #train = data.loc[data.day < 24]
        #test = data.loc[data.day == 24]
    else:
        # 线上提交
        train = data.copy()
        test = pd.read_csv ('/Users/liudong/Desktop/round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = convert_data(test)
        test = deal_category_list(test)
        print ("####################开始测试集中的特征融合#########################")

        # 商品品牌 价格 用户职业 销量
        print ('Merge 商品价格 品牌 用户职业 销量   structure!!!')
        gp = test[['item_price_level', 'item_brand_id', 'user_occupation_id', 'item_sales_level']].groupby (
            by=['item_price_level', 'item_brand_id', 'user_occupation_id'])[
            ['item_sales_level']].std ().reset_index ().rename (index=str,
                                                                columns={'item_sales_level': 'price_occupation'})
        test = test.merge (gp, on=['item_price_level', 'item_brand_id', 'user_occupation_id'], how='left')
        del gp
        gc.collect ()
        # 哑变量
        # data['price_occupation'] = pd.get_dummies (data['price_occupation'])

        # 广告商品的展示时间  商品收藏 商品销量
        print ('Merge  广告商品的展示时间 收藏 销量 structure!!!')
        gp = test[['hour', 'item_collected_level', 'item_sales_level']].groupby (
            by=['hour', 'item_collected_level'])[
            ['item_sales_level']].std ().reset_index ().rename (index=str,
                                                                columns={
                                                                    'item_sales_level': 'context_timestamp_collected'})
        test = test.merge (gp, on=['hour', 'item_collected_level'], how='left')
        del gp
        gc.collect ()
        # list (map (lambda x, y: merge_corr (x, y), data['hour'], data['item_collected_level']))

        # 用户性别和广告商品的品牌
        print ('Merge  用户性别、职业、广告商品的品牌 structure!!!')
        gp = test[['user_gender_id', 'user_occupation_id', 'item_brand_id']].groupby (
            by=['user_gender_id', 'user_occupation_id'])[
            ['item_brand_id']].mean ().reset_index ().rename (index=str,
                                                              columns={'item_brand_id': 'user_sex_brand'})
        test = test.merge (gp, on=['user_gender_id', 'user_occupation_id'], how='left')
        del gp
        gc.collect ()

        # 商品品牌编号—城市编号——销量  var() 方差
        print ('Merge item structure!!!')
        gp = test[['item_brand_id', 'item_city_id', 'item_sales_level']].groupby (by=['item_brand_id', 'item_city_id'])[
            ['item_sales_level']].var ().reset_index ().rename (index=str,
                                                                columns={'item_sales_level': 'item_brand_city_sales'})

        test = test.merge (gp, on=['item_brand_id', 'item_city_id'], how='left')
        del gp
        gc.collect ()

        # 用户性别—年龄—职业-星级 mean()
        print ('Merge user structure!!!')
        gp = test[['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']].groupby (
            by=['user_gender_id', 'user_age_level', 'user_occupation_id'])[
            ['user_star_level']].mean ().reset_index ().rename (index=str, columns={
            'user_star_level': 'user_gender_age_occupation'})
        test = test.merge (gp, on=['user_gender_id', 'user_age_level', 'user_occupation_id'], how='left')
        test['user_gender_age_occupation'] = test['user_gender_age_occupation'].apply ((lambda x: x - 3000))
        del gp
        gc.collect ()

        # 店铺—服务态度—物流-描述-好评率 mean()
        print ('Merge shop structure!!!')
        gp = test[['shop_score_service', 'shop_score_delivery', 'shop_score_description',
                   'shop_review_positive_rate']].groupby (
            by=['shop_score_service', 'shop_score_delivery', 'shop_score_description'])[
            ['shop_review_positive_rate']].std ().reset_index ().rename (index=str,
                                                                         columns={
                                                                             'shop_review_positive_rate': 'shop_service_delivery_review'})
        test = test.merge (gp, on=['shop_score_service', 'shop_score_delivery', 'shop_score_description'], how='left')
        del gp
        gc.collect ()

        # 上下文 展示时间和出现在展示页编号的可能性 mean()
        print ('Merge context structure!!!')

        gp = \
            test[['context_id', 'day', 'item_pv_level', 'hour', 'context_page0']].groupby (
                by=['context_id', 'day', 'item_pv_level', 'hour'])[
                ['context_page0']].mean ().reset_index ().rename (index=str,
                                                                  columns={'context_page0': 'context_day_hour_page'})
        test = test.merge (gp, on=['context_id', 'day', 'item_pv_level', 'hour'], how='left')
        del gp
        gc.collect ()
        print ("####################结束训练集特征融合#########################")

    features = [
            'instance_id',
            # item
            'item_id', 'item_city_id', 'item_price_level',
            'item_sales_level', 'item_collected_level', 'item_pv_level',
            # user
            'user_id', 'user_gender', 'user_age',
            'user_occupation_id', 'user_star',
            # shop
            'shop_id', 'shop_review_num', 'shop_review_positive_rate', 'shop_star_level',
            # context
            'context_id','context_page_id','predict_category_property3','predict_category_property4',
            # 组合特征
            'price_occupation','context_timestamp_collected','user_sex_brand',
            'item_brand_city_sales','user_gender_age_occupation','shop_service_delivery_review',
             'context_day_hour_page',
    ]

    target = ['is_trade']

    if online == False:
        print ('--------Start LGB_Model-------')
        # num_leaves=70, max_depth=8, n_estimators=90, n_jobs=20
        clf = lgb.LGBMClassifier(num_leaves=100, max_depth=9, n_estimators=100, n_jobs=20)
        clf.fit (train[features], train[target], feature_name=features, categorical_feature=['user_gender', ])
        test['lgb_predict'] = clf.predict_proba(test[features], )[:, 1]
        print('LGB出现的误差是：', log_loss(test[target], test['lgb_predict']))
        # 0.0883036  0.082581 0.082731 0.082712 0.0826939 0.08265261
        # 0.08249014913824299
        # 0.08310587911174343 0.08277456472253032 0.08270059707857665
        # 0.08907416747722012 0.08921627287514136
        # 0.07418884461238258 训练集添加is_trade特征的结果最优 出现过拟合状态

    else:

        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[features], train[target], feature_name=features, categorical_feature=['user_gender', ])
        test['predicted_score'] = clf.predict_proba(test[features])[:, 1]
        test[['instance_id', 'predicted_score']].to_csv('baseline.csv', index=False, sep=' ')