import pandas as pd

# 加载训练集
train_data = pd.read_csv('/Users/liudong/Desktop/DonorsChoose/train.csv')
print(train_data.head(10))

# 对训练集进行分析
print(train_data.describe())