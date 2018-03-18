# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# 0. 数据读入及预处理
data_train = pd.read_csv('/Users/liudong/Desktop/titanic/train.csv')


# 1. 去除唯一属性特 axis=1 表示去除PassengerId 和 Ticket列  inplace为真的时候，不返回数据，直接内部执行
data_train.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)

# 2. 类别特征One-Hot编码
data_train['Sex'] = data_train['Sex'].map({'female': 0, 'male': 1}).astype(np.int64)
# 2个Embarked（登录港口地址）缺失值直接填充为S 定位到Embarked为空的值的时候，将值设置为S
data_train.loc[data_train.Embarked.isnull(), 'Embarked'] = 'S'
# 将data_train 和 登录港口进行合并

# contact是起到黏合的作用  get_dummies的作用是将分类变量转换为指示符变量
data_train = pd.concat([data_train, pd.get_dummies(data_train.Embarked)], axis=1)
data_train = data_train.rename(columns={'C': 'Cherbourg','Q': 'Queenstown','S': 'Southampton'})


# 将名字转换
def replace_name(x):
    if 'Mrs' in x: return 'Mrs'
    elif 'Mr' in x: return 'Mr'
    elif 'Miss' in x: return 'Miss'
    else: return 'Other'

data_train['Name'] = data_train['Name'].map(lambda x:replace_name(x))
data_train = pd.concat([data_train, pd.get_dummies(data_train.Name)], axis=1)
data_train = data_train.rename(columns={'Miss': 'Name_Miss','Mr': 'Name_Mr',
                                        'Mrs': 'Name_Mrs','Other': 'Name_Other'})

# print(data_train)

# 3. 数值特征标准化
def fun_scale(df_feature):
    np_feature = df_feature.values.reshape(-1,1).astype(np.float64)
    feature_scale = StandardScaler().fit(np_feature)
    feature_scaled = StandardScaler().fit_transform(np_feature, feature_scale)
    return feature_scale, feature_scaled


Pclass_scale, data_train['Pclass_scaled'] = fun_scale(data_train['Pclass'])
Fare_scale, data_train['Fare_scaled'] = fun_scale(data_train['Fare'])
SibSp_scale, data_train['SibSp_scaled'] = fun_scale(data_train['SibSp'])
Parch_scale, data_train['Parch_scaled'] = fun_scale(data_train['Parch'])

# 4. 缺失值补全及相应处理(使用随机森林方法进行补全未知的年龄值)
# 处理Age缺失值并标准化
# 缺失值处理函数
def set_missing_feature(train_for_missingkey, data, info):
    known_feature = train_for_missingkey[train_for_missingkey.Age.notnull()].as_matrix()
    unknown_feature = train_for_missingkey[train_for_missingkey.Age.isnull()].as_matrix()
    y = known_feature[:, 0] # 第1列作为待补全属性
    x = known_feature[:, 1:] # 第2列及之后的属性作为预测属性
    # 随机森林回归  第一个参数是随机数  第二个参数是在森林中树的个数
    rf = RandomForestRegressor(random_state=0, n_estimators=100)
    # 使用贝叶斯获得的准确度较低
    # rf = BayesianRidge(n_iter=500)
    # 从训练集建立森林
    rf.fit(x, y)
    print(info, "缺失值预测得分", rf.score(x, y))
    predictage = rf.predict(unknown_feature[:, 1:])
    data.loc[data.Age.isnull(), 'Age'] = predictage
    return data


# 处理Cabin(船舱)特征
# 船舱编号为空设置为0，不为空设置为1
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = 1
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = 0
    return df


train_for_missingkey_train = data_train[['Age','Survived','Sex','Name_Miss','Name_Mr','Name_Mrs',
                                         'Name_Other','Fare_scaled','SibSp_scaled','Parch_scaled']]
data_train = set_missing_feature(train_for_missingkey_train, data_train,'Train_Age')
Age_scale, data_train['Age_scaled'] = fun_scale(data_train['Age'])


data_train = set_Cabin_type(data_train)


# 5. 整合数据
train_X = data_train[['Sex','Cabin','Cherbourg','Queenstown','Southampton','Name_Miss','Name_Mr','Name_Mrs','Name_Other',
                      'Pclass_scaled','Fare_scaled','SibSp_scaled','Parch_scaled','Age_scaled']].as_matrix()
train_y = data_train['Survived'].as_matrix()


# 6. 模型搭建及交叉验证
lr = LogisticRegression(C=1.0, tol=1e-6)
svc = SVC(C=1.8, kernel='rbf', decision_function_shape='ovo')
adaboost = AdaBoostClassifier(n_estimators=490, random_state=0)
randomf = RandomForestClassifier(n_estimators=185, max_depth=5, random_state=0)
gbdt = GradientBoostingClassifier(n_estimators=436, max_depth=5, random_state=0)
xgbClassifier = xgb.XGBClassifier(learning_rate = 0.5, n_estimators= 234, max_depth= 6,
                                  min_child_weight= 5, subsample=0.8, colsample_bytree=0.8,
                                  objective= 'binary:logistic', scale_pos_weight=1)

VotingC = VotingClassifier(estimators=[('LR',lr),('SVC',svc),('AdaBoost',adaboost),
                                      ('RandomF',randomf),('GBDT',gbdt),('XGBOOST',xgbClassifier)])

# 模型训练及交叉验证
classifierlist = [('LR',lr),('SVC',svc),('AdaBoost',adaboost),('RandomF',randomf),
                  ('GBDT',gbdt),('XGBOOST',xgbClassifier),('VotingC',VotingC)]
for name, classifier in classifierlist:
    # 分类器训练与下一步交叉验证无关，训练是为下面测试集预测使用
    classifier.fit(train_X, train_y)
    print(name, "Mean_Cross_Val_Score is:",
          cross_val_score(classifier, train_X, train_y, cv=5, scoring='accuracy').mean(), "\n")

# 7. 测试集处理
data_test = pd.read_csv('/Users/liudong/Desktop/titanic/test.csv')
data_test.drop(['Ticket'], axis=1, inplace=True)

data_test['Sex'] = data_test['Sex'].map({'female': 0, 'male': 1}).astype(np.int64)
data_test = pd.concat([data_test, pd.get_dummies(data_test.Embarked)], axis=1)
data_test = data_test.rename(columns={'C': 'Cherbourg','Q': 'Queenstown','S': 'Southampton'})

data_test['Name'] = data_test['Name'].map(lambda x:replace_name(x))
data_test = pd.concat([data_test, pd.get_dummies(data_test.Name)], axis=1)
data_test = data_test.rename(columns={'Miss': 'Name_Miss','Mr': 'Name_Mr',
                                      'Mrs': 'Name_Mrs','Other': 'Name_Other'})
print(data_train)

# 测试集标准化函数
def fun_test_scale(feature_scale, df_feature):
    np_feature = df_feature.values.reshape(-1,1).astype(np.float64)
    feature_scaled = StandardScaler().fit_transform(np_feature, feature_scale)
    return feature_scaled

data_test['Pclass_scaled'] = fun_test_scale(Pclass_scale, data_test['Pclass'])
data_test.loc[data_test.Fare.isnull(),'Fare'] = 0 # 缺失值置为0
data_test['Fare_scaled'] = fun_test_scale(Fare_scale, data_test['Fare'])
data_test['SibSp_scaled'] = fun_test_scale(SibSp_scale, data_test['SibSp'])
data_test['Parch_scaled'] = fun_test_scale(Parch_scale, data_test['Parch'])

# 处理测试集Age缺失值并归一化



train_for_missingkey_test = data_test[['Age','Sex','Name_Miss','Name_Mr','Name_Mrs','Name_Other',
                                       'Fare_scaled','SibSp_scaled','Parch_scaled']]
data_test = set_missing_feature(train_for_missingkey_test, data_test, 'Test_Age')
data_test['Age_scaled'] = fun_test_scale(Age_scale, data_test['Age'])

data_test = set_Cabin_type(data_test)


test_X = data_test[['Sex','Cabin','Cherbourg','Queenstown','Southampton','Name_Miss','Name_Mr','Name_Mrs','Name_Other',
                    'Pclass_scaled','Fare_scaled','SibSp_scaled','Parch_scaled','Age_scaled']].as_matrix()

# 8. 模型预测
model = classifierlist[4]
# 选择分类器
print("Test in %s!" % model[0])
predictions = model[1].predict(test_X).astype(np.int32)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions})
result.to_csv('Result_with_%s.csv' % model[0], index=False)



print("Test in svc")
predictions = svc.predict(test_X).astype(np.int32)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions})
result.to_csv('Result_with_%s.csv' % 'SVC', index=False)



print("Save xgboost!!!")

xgbpred_test = xgbClassifier.predict(test_X).astype(np.int32)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':xgbpred_test})
result.to_csv('Result_with_%s.csv' % 'XGBoost', index=False)


print('...\nAll Finish!')









