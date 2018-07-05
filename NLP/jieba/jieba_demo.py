# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/7/5 下午11:11'

import jieba
# 全模式
seg_list = jieba.cut("7月2日，我开始了在深信服公司为期三个月的实习生活", cut_all=True)
print("Full Mode:" + "/".join(seg_list))

# 精确模式
seg_list = jieba.cut("7月2日，我开始了在深信服公司为期三个月的实习生活", cut_all=False)
print("Default Mode:" + "/".join(seg_list))

# 默认是精确模式
seg_list = jieba.cut("我来到了南山区智园A1栋上班")
print(",".join(seg_list))

# 搜索引擎模式
seg_list = jieba.cut_for_search("我本科毕业于河南财经政法大学，硕士就读于湖南大学信息科学与工程学院")
print(",".join(seg_list))
