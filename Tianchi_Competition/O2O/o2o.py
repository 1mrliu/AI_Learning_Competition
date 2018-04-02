import pandas as pd
import numpy as np
import xgboost as xgb

# 1.数据处理
train_offline = pd.read_csv('/Users/liudong/Desktop/o2o/ccf_offline_stage1_train.csv')
data_test =  pd.read_csv('/Users/liudong/Desktop/o2o/ccf_offline_stage1_test_revised.csv')
all_offline = pd.concat([train_offline, data_test], ignore_index=True)

print(all_offline[10])

# 2.特征分析





# 3.特征工程





# 4.模型设计
# 选用xgboost模型进行判断
xgbClassifier = xgb.XGBClassifier(learning_rate = 0.5, n_estimators= 234, max_depth= 6,
                                  min_child_weight= 5, subsample=0.8, colsample_bytree=0.8,
                                  objective= 'binary:logistic', scale_pos_weight=1)
# xgbClassifier.fit(data_train, data_test)





# 5.对输出结果进行存储
result = pd.DataFrame()
result['User_id'] = data_test['User_id']
result['Coupon_id'] = data_test['Coupon_id']
result['Date_received'] = data_test['Date_received']
print(result)
# result['Probability'] = xgbClassifier.predict()
# result.to_csv('/User/liudong/Desktop/o2o/submission.csv')

