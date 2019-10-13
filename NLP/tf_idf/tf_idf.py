# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2019/10/13 4:08 PM'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 语料
corpus = [
    'This is the first document.',
    'This is the second document',
    'And the third one',
    'is this the first document',
    ]
# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算每个词语出现的次数
X = vectorizer.fit_transform(corpus)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()

# 类
transformer = TfidfTransformer()

# 将词频矩阵统计为TF-IDF值
tfidf = transformer.fit_transform(X)
# 查看数据结构
# tfidf[i][j] 表示i类文本中的tf-idf权重
print(tfidf.toarray())
# tf-idf的矩阵提取出来 元素[i][j] 表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
# 第一个for遍历所有文本
# 第二个for遍历某一类文本下的词语权重
for i in range(len(weight)):
    print("====这输出第",i,"类文本的词语tf-idf权重")
    for j in range(len(word)):
        print(word[j],weight[i][j])