# _*_ coding=utf-8 _*_
import re
r = r'[^/\d]{2,}'
temp  = "13421222fhsjkhfkkjsdfhksh8888asjdgfjsgj432342kjsdfhkhfsk094ashgdjag213543"
temp.replace('2','M')
result = re.findall(r,temp)
print result
print temp

file_dict = {'lisan':12,'lll':15,'dasa':23}
re1 = file_dict['lisan']
print re1
