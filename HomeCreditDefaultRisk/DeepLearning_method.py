# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/6/17 上午 9:00'
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
'''
使用TF来进行操作，实现DNN方法
'''

# 加载数据
sample_size =  None
app_train_df = pd.read_csv('/Users/liudong/Downloads/credit/application_train.csv', nrows=sample_size)
app_test_df = pd.read_csv('/Users/liudong/Downloads/credit/application_test.csv', nrows=sample_size)
bureau_df = pd.read_csv('/Users/liudong/Downloads/credit/bureau.csv', nrows=sample_size)
bureau_balance_df = pd.read_csv('/Users/liudong/Downloads/credit/bureau_balance.csv', nrows=sample_size)
credit_card_df = pd.read_csv('/Users/liudong/Downloads/credit/credit_card_balance.csv', nrows=sample_size)
pos_cash_df = pd.read_csv('/Users/liudong/Downloads/credit/POS_CASH_balance.csv', nrows=sample_size)
prev_app_df = pd.read_csv('/Users/liudong/Downloads/credit/previous_application.csv', nrows=sample_size)
install_df = pd.read_csv('/Users/liudong/Downloads/credit/installments_payments.csv', nrows=sample_size)
print("Main application training data shape= {}".format(app_train_df.shape))
print("Main application test data shape= {}".format(app_test_df.shape))
print("Positive target proportion={}".format(app_train_df['TARGET'].mean()))


def agg_and_merge(left_df, right_df, agg_method, right_suffix):
    """ Aggregate a df by 'SK_ID_CURR' and merge it onto another.
    This method allows feature name """

    agg_df = right_df.groupby ('SK_ID_CURR').agg (agg_method)
    merged_df = left_df.merge (agg_df, left_on='SK_ID_CURR', right_index=True, how='left',
                               suffixes=['', '_' + right_suffix + agg_method.upper ()])
    return merged_df


# 处理数据
def process_dataframe(input_df, encoder_dict=None):
    '''
    Process dataframe into a form useable by LightGBM
    :param input_df: 
    :param encoder_dict: 
    :return: 
    '''
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    categorical_feats = categorical_feats
    encoder_dict = {}
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
        encoder_dict[feat] = encoder
    return input_df, categorical_feats.tolist(), encoder_dict

# feature engineering
def feature_engineering(app_data, bureau_df, bureau_balance_df,credit_card_df,pos_cash_df,prev_app_df,install_df):
    """
    把所有的数据集合并为一个数据表
    """
    #
    app_data['LOAN_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL']
    app_data['ANNUITY_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL']
    # 总体的数量
    app_data['ANNUITY LENGTH'] = app_data['AMT_CREDIT'] / app_data['AMT_ANNUITY']
    # 社会特征
    app_data['WORKING_LIFE_RATIO'] = app_data['DAYS_EMPLOYED'] / app_data['DAYS_BIRTH']
    app_data['INCOME_PER_FAM'] = app_data['AMT_INCOME_TOTAL'] / app_data['CNT_FAM_MEMBERS']
    app_data['CHILDREN_RATIO'] = app_data['CNT_CHILDREN'] / app_data['CNT_FAM_MEMBERS']
    # 用nan取代数字的值
    prev_app_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)

    # Previous applications
    print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))
    agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}
    prev_apps = prev_app_df.groupby('SK_ID_CURR').agg(agg_funs)
    prev_apps.columns = ['PREV APP COUNT', 'TOTAL PREV LOAN AMT']
    merged_df = app_data.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')
    # Average the rest of the previous app data
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, prev_app_df, agg_method, 'PRV')
    print ('Shape after merging with previous apps num data = {}'.format(merged_df.shape))
    # Previous app categorical features
    prev_app_df, cat_feats, _ = process_dataframe(prev_app_df)
    prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR') \
        .agg ({k: lambda x: str (x.mode ().iloc[0]) for k in cat_feats})
    merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
                                 how='left', suffixes=['', '_BAVG'])
    print ('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))
    # Credit card data - numerical features
    wm = lambda x: np.average (x, weights=-1 / credit_card_df.loc[x.index, 'MONTHS_BALANCE'])
    credit_card_avgs = credit_card_df.groupby ('SK_ID_CURR').agg (wm)
    merged_df = merged_df.merge (credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
                                 how='left', suffixes=['', '_CC_WAVG'])
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge (merged_df, credit_card_avgs, agg_method, 'CC')
    print ('Shape after merging with previous apps num data = {}'.format (merged_df.shape))

    # Credit card data - categorical features
    most_recent_index = credit_card_df.groupby ('SK_ID_CURR')['MONTHS_BALANCE'].idxmax ()
    cat_feats = credit_card_df.columns[credit_card_df.dtypes == 'object'].tolist () + ['SK_ID_CURR']
    merged_df = merged_df.merge (credit_card_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR',
                                 right_on='SK_ID_CURR',
                                 how='left', suffixes=['', '_CCAVG'])
    print ('Shape after merging with credit card data = {}'.format (merged_df.shape))

    # Credit bureau data - numerical features
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge (merged_df, bureau_df, agg_method, 'B')
    print ('Shape after merging with credit bureau data = {}'.format (merged_df.shape))

    # Bureau balance data
    most_recent_index = bureau_balance_df.groupby ('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax ()
    bureau_balance_df = bureau_balance_df.loc[most_recent_index, :]
    merged_df = merged_df.merge (bureau_balance_df, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU',
                                 how='left', suffixes=['', '_B_B'])
    print ('Shape after merging with bureau balance data = {}'.format (merged_df.shape))

    # Pos cash data - weight values by recency when averaging
    wm = lambda x: np.average (x, weights=-1 / pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])
    f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF': wm}
    cash_avg = pos_cash_df.groupby ('SK_ID_CURR')['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE',
                                                  'SK_DPD', 'SK_DPD_DEF'].agg (f)
    merged_df = merged_df.merge (cash_avg, left_on='SK_ID_CURR', right_index=True,
                                 how='left', suffixes=['', '_CAVG'])

    # Unweighted aggregations of numeric features
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge (merged_df, pos_cash_df, agg_method, 'PC')

    # Pos cash data data - categorical features
    most_recent_index = pos_cash_df.groupby ('SK_ID_CURR')['MONTHS_BALANCE'].idxmax ()
    cat_feats = pos_cash_df.columns[pos_cash_df.dtypes == 'object'].tolist () + ['SK_ID_CURR']
    merged_df = merged_df.merge (pos_cash_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR',
                                 right_on='SK_ID_CURR',
                                 how='left', suffixes=['', '_CAVG'])
    print ('Shape after merging with pos cash data = {}'.format (merged_df.shape))

    # Installments data
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge (merged_df, install_df, agg_method, 'I')
    print ('Shape after merging with installments data = {}'.format (merged_df.shape))

    # Add more value counts
    merged_df = merged_df.merge (pd.DataFrame (bureau_df['SK_ID_CURR'].value_counts ()), left_on='SK_ID_CURR',
                                 right_index=True, how='left', suffixes=['', '_CNT_BUREAU'])
    merged_df = merged_df.merge (pd.DataFrame (credit_card_df['SK_ID_CURR'].value_counts ()), left_on='SK_ID_CURR',
                                 right_index=True, how='left', suffixes=['', '_CNT_CRED_CARD'])
    merged_df = merged_df.merge (pd.DataFrame (pos_cash_df['SK_ID_CURR'].value_counts ()), left_on='SK_ID_CURR',
                                 right_index=True, how='left', suffixes=['', '_CNT_POS_CASH'])
    merged_df = merged_df.merge (pd.DataFrame (install_df['SK_ID_CURR'].value_counts ()), left_on='SK_ID_CURR',
                                 right_index=True, how='left', suffixes=['', '_CNT_INSTALL'])
    print ('Shape after merging with counts data = {}'.format (merged_df.shape))

    return merged_df

# 合并数据集为一个单独的训练集
len_train = len(app_train_df)
app_both = pd.concat([app_train_df,app_test_df])
merge_df = feature_engineering(app_both, bureau_df,bureau_balance_df,credit_card_df,pos_cash_df,prev_app_df,
                               install_df)
# 分离metadata
meta_cols = ['SK_ID_CURR', 'SK_ID_BUREAU','SK_ID_PREV']
meta_df = merge_df[meta_cols]
merge_df.drop(meta_cols, axis=1, inplace=True)

# 处理数据集
merge_df, categorical_feats, encoder_dict = process_dataframe(input_df=merge_df)
# Capture other categorical features not as object data types:
non_obj_categoricals = [
    'FONDKAPREMONT_MODE',
    'HOUR_APPR_PROCESS_START',
    'HOUSETYPE_MODE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'NAME_INCOME_TYPE',
    'NAME_TYPE_SUITE',
    'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE',
    'WALLSMATERIAL_MODE',
    'WEEKDAY_APPR_PROCESS_START',
    'NAME_CONTRACT_TYPE_BAVG',
    'WEEKDAY_APPR_PROCESS_START_BAVG',
    'NAME_CASH_LOAN_PURPOSE',
    'NAME_CONTRACT_STATUS',
    'NAME_PAYMENT_TYPE',
    'CODE_REJECT_REASON',
    'NAME_TYPE_SUITE_BAVG',
    'NAME_CLIENT_TYPE',
    'NAME_GOODS_CATEGORY',
    'NAME_PORTFOLIO',
    'NAME_PRODUCT_TYPE',
    'CHANNEL_TYPE',
    'NAME_SELLER_INDUSTRY',
    'NAME_YIELD_GROUP',
    'PRODUCT_COMBINATION',
    'NAME_CONTRACT_STATUS_CCAVG',
    'STATUS',
    'NAME_CONTRACT_STATUS_CAVG'
]
categorical_feats = categorical_feats + non_obj_categoricals
# 将 target 从数据集中去除  为了scaling
labels = merge_df.pop('TARGET')
labels = labels[:len_train]
# reshape (one-hot)
target = np.zeros([len(labels), len(np.unique(labels))])
target[:, 0] = labels == 0
target[:, 1] = labels == 1
# 对数据集中的空数据进行处理
null_counts = merge_df.isnull().sum()
null_counts = null_counts[null_counts > 0]
null_ratios = null_counts / len(merge_df)
# 删除超过 x% 的空的数据
null_thresh = .8
null_cols = null_ratios[null_ratios > null_thresh].index
merge_df.drop(null_cols, axis=1, inplace=True)
print('Columns dropped for being over {}% null:'.format(100*null_thresh))
for col in null_cols:
    print(col)
    if col in categorical_feats:
        categorical_feats.pop(col)

# 对分类的数据进行处理
cat_feats_idx = np.array([merge_df.columns.get_loc(x) for x in categorical_feats])
int_feats_idx = [merge_df.columns.get_loc(x) for x in non_obj_categoricals]
cat_feats_lookup = pd.DataFrame({'feature': categorical_feats, 'columns_index': cat_feats_idx})
cat_feats_lookup.head()

cont_feats_idx = np.array([merge_df.columns.get_loc(x)
                           for x in merge_df.columns[-merge_df.columns.isin(categorical_feats)] ])
cont_feats_lookup = pd.DataFrame(
    {'feature':merge_df.columns[-merge_df.columns.isin(categorical_feats)],
     'columns_index':cont_feats_idx}
)
cont_feats_lookup.head()

# 对特征进行缩放 避免进行不同程度的加权
scaler = StandardScaler()
final_col_names = merge_df.columns
merge_df = merge_df.values
merge_df[:,cont_feats_idx] = scaler.fit_transform(merge_df[:, cont_feats_idx])

scaler2 = MinMaxScaler(feature_range=(0,1))
merge_df[:, int_feats_idx] = scaler2.fit_transform(merge_df[:, int_feats_idx])

# 将数据分为有标记和无标记的数据集
train_df = merge_df[:len_train]
predict_df = merge_df[len_train:]
del merge_df, app_train_df, app_test_df, bureau_df,bureau_balance_df, credit_card_df,pos_cash_df, prev_app_df

# 验证集
X_train, X_valid, y_train, y_valid = train_test_split(train_df, target, test_size=0.1, random_state=2,
                                                      stratify=target[:, 0])
# 构造NN的输入图结构
# graph的参数设置
N_HIDDEN_1 = 80
N_HIDDEN_2 = 80
N_HIDDEN_3 = 40
n_cont_inputs = X_train[:, cont_feats_idx].shape[1]
n_classes = 2

# 学习率参数
LEARNING_RATE = 0.01
N_EPOCHS = 30
N_ITERATIONS = 400
BATCH_SIZE = 250

print('Number of continous features:', n_cont_inputs)
print('Number of categoricals pre-embedding:', X_train[:, cat_feats_idx].shape[1])

def embed_and_attach(X, X_cat, cardinality):
    embedding = tf.Variable(tf.random_uniform([cardinality, cardinality // 2], -1.0, 1.0))
    embedded_x = tf.nn.embedding_lookup(embedding, X_cat)
    return tf.concat([embedded_x, X], axis=1)

tf.reset_default_graph()

# 为categorical variables  定义placeholder
cat_placeholders, cat_cardinalities = [], []
for idx in cat_feats_idx:
    exec ('X_cat_{} = tf.palceholder(tf.int32,shape=(None,), name=\'X_cat_{}\''.format(idx,idx))
    exec ('cat_placeholders.append(X_cat_{})'.format(idx))
    cat_cardinalities.append(len(np.unique(np.concatenate([train_df[:, idx],predict_df[:, idx]],axis=0))))

# other placeholders
X_cont = tf.placeholder(tf.float32, shape=(None, n_cont_inputs), name='X_cont')
y = tf.placeholder(tf.int32, shape=(None, n_classes), name='labels')
train_mode = tf.placeholder(tf.bool)

# Add embeddings to input
X = tf.identity(X_cont)
for feat, card in zip(cat_placeholders, cat_cardinalities):
    X = embed_and_attach(X, feat, card)

# 定义网络层的数据
# 对于过拟合使用L2正则化进行处理
with tf.name_scope('dnn'):
    hidden_layer_1 = tf.layers.dense(inputs=X,
                                     units=N_HIDDEN_1,
                                     name='first_hidden_layer',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.3))
    hidden_layer_1 = tf.layers.batch_normalization(hidden_layer_1, training=train_mode)
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)

    drop_layer_1 = tf.layers.dropout(inputs=hidden_layer_1,
                                     rate=0.4,
                                     name='first_dropout_layer',
                                     training=train_mode)

    hidden_layer_2 = tf.layers.dense(inputs=drop_layer_1,
                                     units=N_HIDDEN_2,
                                     name='second_hidden_layer',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
    hidden_layer_2 = tf.layers.batch_normalization(hidden_layer_2, training=train_mode)
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)

    drop_layer_2 = tf.layers.dropout(inputs=hidden_layer_2,
                                     rate=0.2,
                                     name='second_dropout_layer',
                                     training=train_mode)

    hidden_layer_3 = tf.layers.dense(inputs=drop_layer_2,
                                     units=N_HIDDEN_3,
                                     name='third_hidden_layer')
    hidden_layer_3 =  tf.layers.batch_normalization(hidden_layer_3, training=train_mode)
    hidden_layer_3 = tf.nn.relu(hidden_layer_3)

    logits = tf.layers.dense(inputs=hidden_layer_3,
                            units=n_classes,
                             name='outputs')

# 将交叉熵定义为损失函数
with tf.name_scope('loss'):
    xent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logits)
    loss = tf.reduce_mean(xent, name='loss')
# 定义优化器
with tf.name_scope('train'):
    optimiser = tf.train.AdamOptimizer()
    train_step = optimiser.minimize(loss)
# 评价  AUC
with tf.name_scope('eval'):
    predict = tf.argmax(logits, axis=1, name='class_predictions')
    predict_proba = tf.nn.softmax(logits, name='probability_predictions')
# 初始化节点并保存
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def get_feed_dict(cat_feats_idx, cat_placeholders, cont_feats_idx, batch_X, batch_y=None,
                  training=False):
    '''
    return a feed dict for the graph including all the categorical fetures to embed
    '''
    feed_dict = {X_cont:batch_X[:, cont_feats_idx]}
    if batch_y is not None:
        feed_dict[y] = batch_y

    # Loop through the categorical features to provide values for the palceholders
    for idx, tensor in zip(cat_feats_idx,cat_placeholders):
        feed_dict[tensor] = batch_X[:, idx].reshape(-1, ).astype(int)

    # Training mode or not
    feed_dict[train_mode] = training

    return feed_dict

train_auc, valid_auc = [], []
n_rounds_not_improved = 0
early_stopping_epochs = 2
with tf.Session() as sess:
    init.run()
    # begin epoch loop
    print('Training for {} iterations over {} epochs with batchsize {] ...'.format(N_ITERATIONS, N_EPOCHS,BATCH_SIZE))
    for epoch in range(N_EPOCHS):
        for iteration in range(N_ITERATIONS):
            # Get random selection of data for batch GD. Upsample positive classes to make it
            # balanced in the training batch
            pos_ratio = 0.5
            pos_idx = np.random.choice(np.where(y_train[:, 1] == 1)[0],size=int(np.round(BATCH_SIZE*pos_ratio)))
            neg_idx = np.random.choice(np.where(y_train[:, 1] == 0)[0],size=int(np.round(BATCH_SIZE*(1-pos_ratio))))
            idx = np.concatenate([pos_idx, neg_idx])

            # run training
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            sess.run([train_step, extra_update_ops],
                     feed_dict=get_feed_dict(cat_feats_idx,cat_placeholders,cont_feats_idx,
                                             X_train[idx,:], y_train[idx,:], 1))
        # AUC
        y_pred_train, y_prob_train =sess.run([predict, predict_proba],
                                         feed_dict=get_feed_dict(cat_feats_idx,cat_placeholders,cont_feats_idx,
                                                                 X_train,y_train,False))
        train_auc.append(roc_auc_score(y_train[:, 1], y_prob_train[:,1]))
        y_pred_val, y_prob_val = sess.run([predict,  predict_proba],
                                      feed_dict=get_feed_dict(cat_feats_idx,cat_placeholders,cont_feats_idx,
                                                              X_valid,y_valid, False))
        valid_auc.append(roc_auc_score(y_valid[:,1], y_prob_val[:,1]))

        # early stopping
        if epoch > 1:
           best_epoch_so_far = np.argmax(valid_auc[:-1])
           if valid_auc[epoch] <= valid_auc[best_epoch_so_far]:
              n_rounds_not_improved += 1
           else:
              n_rounds_not_improved = 0
           if n_rounds_not_improved > early_stopping_epochs:
              print('Early stopping due to no improvement after {} epochs'.format(early_stopping_epochs))
              break
        print('Epoch = {}, Train AUC = {:.8f}, Valid AUC = {:.8f}'.format(epoch, train_auc[epoch],valid_auc[epoch]))
    # once trained make prediction
    print('Training complete')
    y_prob = sess.run(predict_proba,feed_dict=get_feed_dict(
        cat_feats_idx, cat_placeholders,cont_feats_idx, predict_df, None, False))

out_df = pd.DataFrame({'SK_ID_CURR': meta_df['SK_ID_CURR'][len_train:], 'TARGET': y_prob[:, 1]})
out_df.to_csv('nn_submission.csv', index=False)
