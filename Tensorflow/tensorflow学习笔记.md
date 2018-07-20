## 1.logits参数的介绍
### tensorflow中求交叉熵的函数中有logits变量：
```
# 计算交叉熵的函数
tf.nn.softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None
)
```
#### logits本身是一个函数，它可以把某个概率从[0,1]映射为 [-inf,+inf] (正负无穷)，函数化描述形式为：logits=ln(p/(1-p))。因此我们可以把logits视为原生态的、未经缩放的，可视为一种未归一化的概率，例如[4,1,2]。softmax的作用就是，把一个系数从[-inf,+inf]映射到[0,1]，同时所有参与映射的值累计之和等于一，变成[0.95,0.05,0]。经过softmax就可以当做概率来用。logits作为softmax的输入，经过softmax的加工，就变成归一化的概率，然后和labels代表的概率分布进行计算之间的交叉熵。
```
import tensorflow as tf
 
labels = [[0.2,0.3,0.5],
          [0.1,0.6,0.3]]
logits = [[4,1,-2],
          [0.1,1,3]]
 
logits_scaled = tf.nn.softmax(logits)
# 已经淘汰 result = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
with tf.Session() as sess:
    print (sess.run(logits_scaled))
    print (sess.run(result))
```
运行结果：
```
[[0.95033026 0.04731416 0.00235563]
 [0.04622407 0.11369288 0.84008306]]
[3.9509459 1.6642545]
```
### 注意事项：
1.如果labels的每一行是one-hot表示，也就是只有一个地方为1（或者说100%），其他地方为0（或者说0%），还可以使用tf.sparse_softmax_cross_entropy_with_logits()。之所以用100%和0%描述，就是让它看起来像一个概率分布。
2.参数labels,logits必须有相同的形状 [batch_size, num_classes] 和相同的类型(float16, float32, float64)中的一种，否则交叉熵无法计算。
