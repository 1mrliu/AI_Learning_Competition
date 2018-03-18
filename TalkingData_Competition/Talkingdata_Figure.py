"""
# 特征分析（统计学和绘图）

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/Users/liudong/Desktop/talkingdata/train_sample.csv')
data.drop(['attributed_time'], axis=1, inplace=True)
# 查看设备的使用次数 和 对应的操作系统的种类
device_data = data['device'].value_counts()
os_data = data['os'].value_counts()
# 计算相关性协方差  corr()函数， 返回结果接近0说明无相关，大于0说明是正相关，小于0是负相关
data_corr = data.corr()
# 画出特征相关性的热力图
sns.set()
a = plt.figure('特征相关性热力图')
ax = plt.subplot(111)
heat_map = sns.heatmap(data_corr, vmin=-1, vmax=1, annot=True, square=True)
plt.plot(10,6)
plt.show()
plt.close()

# 通过上边的热力图发现ip和is_attributed之间关系比较大
data[['ip', 'is_attributed']].groupby(['ip'])




print(data[:10])
print(data_corr)
