# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/4/23 下午2:28'
#import some necessary librairies

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
# ignore annoying warning (from sklearn and seaborn)
warnings.warn = ignore_warn
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# 设置输出的浮点数的格式
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

from subprocess import check_output
# check the files available in the directory
# print(check_output(["ls", "/Users/liudong/Desktop/house_price/train.csv"]).decode("utf8"))
# 加载数据
train = pd.read_csv('/Users/liudong/Desktop/house_price/train.csv')
test = pd.read_csv('/Users/liudong/Desktop/house_price/test.csv')
# 查看训练数据的特征
print(train.head(5))
# 查看测试数据的特征
print(test.head(5))

# 查看未删除ID之前数据的shape （1460， 81） （1459， 80）
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

# 保存 Id列的值
train_ID = train['Id']
test_ID = test['Id']

# 删除ID列的值，因为它对于预测结果没有太大的影响
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# 检查删除ID以后的数据的shape （1460， 80） （1459， 79）
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))

# 删除那些异常数据值   异常值的处理对应于那些极端情况
'''
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
这里举的是异常值的处理  Example：房子面积很大，但是价格很低 其他的这种异常并不是需要全部删除
还要保持模型的健壮性
'''
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# 对SalePrice使用log(1+x)的形式来处理
train["SalePrice"] = np.log1p(train["SalePrice"])


# 特征工程
# 将train数据集和test数据集结合在一起
# ntrain ntest 各自行数
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
# contact连接以后index只是重复，会出现逻辑错误  需要使用reset_index处理 drop为True
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

# 特征工程  对特征进行处理
# 处理缺失数据
print(all_data.isnull().sum())
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(20))

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
# all records are "AllPub", except for one "NoSeWa" and 2 NA  remove
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
# 检查是否还有缺失值存在 输出结果是没有缺失值存在
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head())
# 附加的特征工程
# 把一些数值变量分类
#MSSubClass=建筑物的类别
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# 这些类别型的数据需要转换为数值型的数据
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# 将这些有类别的特征值(例如英文表示的额) 使用LabelEncoder变成分类的数值数据
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))


# shape
print('Shape all_data: {}'.format(all_data.shape))

# 增加更多重要的特征
# 计算出房子总的面积  包含上下两层的面积 基础的面积  一层面积 二层面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# Skewed features 将内容为数值型的特征列找出来  skewed为偏斜度的设置
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# 检查所有数值特征的偏斜度
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False) #降序排列
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))

# Box Cox Transformation of (highly) skewed features
# We use the scipy function boxcox1p which computes the Box-Cox transformation of  1+x .
# Note that setting  λ=0  is equivalent to log1p used above for the target variable.
skewness = skewness[abs (skewness) > 0.75]
print ("There are {} skewed numerical features to Box Cox transform".format (skewness.shape[0]))

# 转换数据位正态分布，避免长尾分布现象的出现
from scipy.special import boxcox1p

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    # all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
# one-hot编码
all_data = pd.get_dummies(all_data)
print(all_data.shape)
'''
###########################
对数据处理以后新的训练集和测试集
###########################
'''
train = all_data[:ntrain]
test = all_data[ntrain:]
# train, test = train_test_split(all_data, train_size=0.8, random_state=2)

#k-fold交叉验证 5折交叉验证
n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return(rmse)

# 模型的选择和使用
# LASSO Regression :
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
# Elastic Net Regression 弹性网回归
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
# Kernel Ridge Regression 岭回归 alpha超参数
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# Gradient Boosting Regression 梯度增强回归
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
#  XGboost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
# lightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# 基础模型的评价得分  mean计算均值   std() 计算的是标准偏差
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# 基础模型的分类器的平均值
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    # 模型的fit()函数
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    # 模型的预测 k-fold为预测的次数，将多列的数据合并，并取平均值
    def predict(self, X):
        # column_stack可以将多个列组合   [0,1],[2,3] 组合为 [[0,2],[1,3]]
        predictions = np.column_stack([model.predict (X) for model in self.models_])
        return np.mean(predictions, axis=1)
# 评价这四个模型的好坏
averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
score = rmsle_cv(averaged_models)
print (" Averaged base models score: {:.4f} ({:.4f})\n".format (score.mean (), score.std()))


# 增加Meta-model 最最精彩的部分***回归预测***
class StackingAveragedModels (BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list () for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # 使用K-fold的方法来进行交叉验证，将每次验证的结果作为新的特征来进行处理
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index],  y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 将交叉验证预测出的结果 和 训练集中的标签值进行训练
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # 从得到的新的特征  采用新的模型进行预测  并输出结果
    def predict(self, X):
        meta_features = np.column_stack ([
            np.column_stack([model.predict (X) for model in base_models]).mean (axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), # meta_model=model_lgb)
                                                 meta_model=lasso)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# 均方根误差评价函数
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# 最后的Training、Prediction 输出结果
# StackedRegressor
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

# XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

# lightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))

'''
RMSE on the entire Train data when averaging
'''
print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 + xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
# 模型融合的预测效果
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
# 保存结果
result = pd.DataFrame()
result['Id'] = test_ID
result['SalePrice'] = ensemble
# index=False 是用来除去行编号
result.to_csv('/Users/liudong/Desktop/house_price/result.csv', index=False)
print('##########结束训练##########')