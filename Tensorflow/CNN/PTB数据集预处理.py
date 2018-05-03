# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/5/3 下午4:59'
"""
PTB数据集预处理，生成词汇表
"""
import codecs
import collections
from operator import itemgetter

# 训练集数据文件
RAW_DATA = "/Users/liudong/Desktop/simple-examples/data/ptb.train.txt"
# 输出的词汇表文件
VOCAB_OUTPUT = "ptb.vocab"
# 统计单词出现的频率
counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

# 按照词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

# 稍后会在文本换行处加上句子结束符"<eos>" 这里预先将它加入词汇表
sorted_words = ["eos"] + sorted_words

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")