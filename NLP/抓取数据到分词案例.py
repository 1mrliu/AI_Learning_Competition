# _*_ coding:utf-8 _*_
from bs4 import BeautifulSoup
from urllib import request
import re 
import jieba
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (10.0,5.0)
# from wordcloud import WordCloud 



resp = request.urlopen('https://movie.douban.com/nowplaying/hangzhou/')
html_data = resp.read().decode('utf-8')
# print(html_data)

soup = BeautifulSoup(html_data,'html.parser')
# print(soup)
nowplaying_movie = soup.find_all('div', class_='mod-bd')
nowplaying_movie_list = nowplaying_movie[0].find_all('li', class_='list-item')
# print(nowplaying_movie_list[0])
nowplaying_list = []
for item in nowplaying_movie_list:
    nowplaying_dict = {}
    nowplaying_dict['id'] = item['data-subject']
    for tag_img_item in item.find_all('img'):
        nowplaying_dict['name'] = tag_img_item['alt']
        nowplaying_list.append(nowplaying_dict)

print(nowplaying_list)

# 对电影的短评进行处理
requrl = "https://movie.douban.com/subject/" + nowplaying_list[0]['id'] + '/comments'+ '?' + 'staus=p' 
resp = request.urlopen(requrl)
html_data = resp.read().decode('utf-8')
soup = BeautifulSoup(html_data,'html.parser')
comment_div_lits = soup.find_all('div', class_='comment')
# print(comment_div_lits)

eachCommentList = []
for item in comment_div_lits:
    if item.find_all('span',class_="short")[0].string is not None:
        eachCommentList.append(item.find_all('span',class_="short")[0].string)
print(eachCommentList)

# 清理数据
comments = ''
for k in  range(len(eachCommentList)):
    comments = comments + (str(eachCommentList[k])).strip()

pattern = re.compile(r'[\u4e00-\u9fa5]+')
filterdata = re.findall(pattern,comments)
cleaned_comments = ''.join(filterdata)
print(cleaned_comments)

# 使用jieba进行分词
segment = jieba.lcut(cleaned_comments)
words_df = pd.DataFrame({'segment':segment})
print(words_df.head())

# 去除停用词
stopwords = pd.read_csv('D:/Users/sangfor/Desktop/chineseStopWords.txt',sep="\t",names=['stopword'], encoding='gb2312')
words_df = words_df[~words_df.segment.isin(stopwords.stopword)]
print(words_df.head())

# 词频统计
words_stat =  words_df.groupby(by=['segment'])['segment'].agg({"计数":np.size})
words_stat = words_stat.reset_index().sort_values(by=['计数'],ascending=False)

# 码云统计 
# 其中simhei.ttf使用来指定字体的，
# 可以在百度上输入simhei.ttf进行下载后，放入程序的根目录即可。
