import time
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

path_test_out = "model/"
trains = pd.read_csv('/Users/liudong/Desktop/PINGAN-2018-train_demo.csv', iterator=True)
loop = True
chunkSize = 10000
chunks = []
while loop:
    try:
        chunk = trains.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
train = pd.concat(chunks, ignore_index=True)



# data = pd.read_csv('/Users/liudong/Desktop/PINGAN-2018-train_demo.csv')
# # 计算相关性协方差  corr()函数， 返回结果接近0说明无相关，大于0说明是正相关，小于0是负相关
# data_corr = data.corr()
# # 画出特征相关性的热力图
# sns.set()
# a = plt.figure('特征相关性热力图')
# ax = plt.subplot(111)
# heat_map = sns.heatmap(data_corr, vmin=-1, vmax=1, annot=True, square=True)
# plt.plot(10,6)
# plt.show()
# plt.close()


# 定义处理时间的函数
# df2["start_time"]  = list(map(lambda t: datetime.datetime.date(t),  df2["start_time"]))
def dataPreProcessTime(data):
    format = '%Y-%m-%d %H:%M:%S'
    # value为传入的值为时间戳(整形)，如：1332888820
    data['TIME'] = time.localtime (data['TIME'])
    ## 经过localtime转换后变成
    ## time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=0)
    # 最后再经过strftime函数转换为正常日期格式。
    data['TIME'] = time.strftime (format, data['TIME'])

    data['TIME'] = pd.to_datetime(data['TIME']).dt.date
    data['TIME'] = data['TIME'].apply(lambda x: x.strftime('%Y%m%d')).astype('int')
    return data



# train = dataPreProcessTime(train)
# print(train)
xgbClassifier = xgb.XGBClassifier (learning_rate=0.5,
                                   n_estimators=234,
                                   max_depth=6,
                                   min_child_weight=5,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   objective='binary:logistic',
                                   scale_pos_weight=1)
train_X = train[['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE']]
train_Y = train['Y']
train.drop(['Y'], axis=1, inplace=True)

print('Start practise!!!')

xgbClassifier.fit(train_X, train_Y)

result = pd.DataFrame({'Id':train['TERMINALNO']})

result['Pred'] = xgbClassifier.predict(train)
# inplace为正，可以删除重复值
result.drop_duplicates('Id', inplace=True)

result.to_csv(path_test_out+'result.csv', index=False)
print("Output result!!!")
