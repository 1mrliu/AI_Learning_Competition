import pandas as pd
import numpy as np

# 用户信息
unames = ['user_id','gender','age','occupation','zip']
# seq代表的是将数据按照连接符号进行分割 names= 是按照unames的顺序对输出的结果的列进行列名的
users = pd.read_table('/Users/liudong/Desktop/ml-1m/users.dat', sep='::', header=None, names=unames)

# 评分
rname = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('/Users/liudong/Desktop/ml-1m/ratings.dat', sep="::", header=None, names=rname)

# 电影信息
mname = ['movie_id', 'title', 'genres']
movies = pd.read_table('/Users/liudong/Desktop/ml-1m/movies.dat', sep='::', header=None, names=mname)

# print(users[:5])
# print(ratings)
# print(movies[:5])
# 将几个表的数据进行结合，相同的列会进行合并
data = pd.merge(pd.merge(ratings, users),movies)
print(data[:5])
#
mean_ratings = pd.pivot_table(data, index='title', values='rating', columns='gender', aggfunc='mean')
print(mean_ratings[:5])

# 获取电影名字的分组大小
ratings_by_title = data.groupby('title').size()
# print(ratings_by_title[:10])

# 将电影分组中数据大于250的数据提取出来
active_titles =  ratings_by_title.index[ratings_by_title >= 250]
# print(active_titles[:10])


mean_ratings = mean_ratings.ix[active_titles]
print(mean_ratings[:3])


# 女性最喜欢的电影降序排列
top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)
print(top_female_ratings)

# 男女性之间差异最大的电影
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
print(sorted_by_diff[:])

# 对行进行反序，并取出前十五行
print(sorted_by_diff[::-1][:15])

# 根据电影名称分组的得分数据的标准差
rating_std_by_title = data.groupby('title')['rating'].std()

# 根据active_titles 进行过滤
rating_std_by_title = rating_std_by_title.ix[active_titles]


# 根据值对Series进行降序排列
print(rating_std_by_title[:10])

print(rating_std_by_title.sort())






