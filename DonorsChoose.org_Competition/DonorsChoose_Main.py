import pandas as pd

# 加载训练集
train_data = pd.read_csv('/Users/liudong/Desktop/DonorsChoose/train.csv')
print(train_data)

# 加载测试集
test_data  = pd.read_csv('/Users/liudong/Desktop/DonorsChoose/test.csv')
print(test_data)

# 加载资源集
resource_data = pd.read_csv('/Users/liudong/Desktop/DonorsChoose/resources.csv')
print(resource_data)