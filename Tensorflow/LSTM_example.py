# coding:utf-8
import time
from collections import namedtuple
import tensorflow as tf
import numpy as np 

# 读取数据预处理
with open('D:/Users/sangfor/Desktop/anna.txt', 'r') as f:
    text = f.read()
# print(text)
vocab = set(text)
# print(vocab)
vocab_to_int = {c:i for i,c in enumerate(vocab)}
print(vocab_to_int)
int_to_vocab = dict(enumerate(vocab))
# print(int_to_vocab)
# 将txt文件转化为数字编号的文件
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

# 分割batch
def get_batches(arr,n_seqs,n_steps):
    '''
    arr:数组
    n_seqs:一个batch中的序列个数
    n_steps:单个序列包含的字符
    '''
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]

    # 重塑
    arr = arr.reshape((n_seqs,-1))
    for n in range(0,arr.shape[1],n_steps):
        # inputs
        x = arr[:, n:n+n_steps]
        # targets
        y = np.zeros_like(x)
        y[:,:-1], y[:,-1] = x[:,1:], x[:,0]
        yield x, y
batches = get_batches(encoded,10,50)
x, y = next(batches)

print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])  

# 构建输入层
def build_inputs(num_seqs,num_steps):
    '''
    num_seqs:每个batch中的序列个数
    num_steps:每个序列包含的字符数
    '''
    inputs = tf.placeholder(tf.int32,shape=(num_seqs,num_steps),name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs,num_steps),name='targets')

    # keep_prob dropout率
    keep_prob = tf.placeholder(tf.int32, name='keep_prob')

    return inputs, targets, keep_prob
# LSTM层
def build_lstm(lstm_size, num_layers,batch_size,keep_prob):
    '''
    lstm_size:隐藏层中的节点数
    num_layers:隐藏层的个数
    batch_size
    '''
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=keep_prob)
    # 堆叠
    cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size,tf.float32) 
    return cell, initial_state

# 输出层
def build_output(lstm_output, in_size,out_size):
    '''
    输出结果
    输出层重塑后的size
    softmax层的size
    '''
    seq_output = tf.concat(1,lstm_output)
    x = tf.reshape(seq_output,[-1,in_size])

    # 将lstm层和softmax层进行连接
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size],stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size)) 
    # logits
    logits = tf.matmul(x,softmax_w) + softmax_b

    out = tf.nn.softmax(logits,name='predictions')
    return out, logits

# 训练误差计算
def build_loss(logits,targets,lstm_size,num_classes):
    '''
    logits是全连接层的输出结果--没有经过softmax
    num_classes vocab_size
    '''
    y_one_hot = tf.one_hot(targets,num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.getshape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    return loss

# Optimizer 优化
def build_optimizer(loss, learning_rate, grad_clip):
    # 使用clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss,tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer

# 模型组合
# 使用tf.nn.dynamic_run来运行RNN序列
class CharRNN:
    def __init__(self,num_classes,batch_size=64,num_steps=50,
                      lstm_size=128,num_layers=2,learning_rate=0.001,
                      grad_clip=5,sampling=False):
        if sampling==True:
            batch_size,num_steps = 1,1
        else:
            batch_size,num_steps = batch_size, num_steps
        tf.reset_default_graph()
        # 输入层
        self.inputs, self.targets,self.keep_prob = build_inputs(batch_size,num_steps)
        # LSTM层
        cell, self.initial_state = build_lstm(lstm_size,num_layers,batch_size,self.keep_prob)

        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, num_classes)

        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        

        # Loss  optimizer
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss,learning_rate,grad_clip)
# 模型训练
# - num_seqs: 单个batch中序列的个数
# - num_steps: 单个序列中字符数目
# - lstm_size: 隐层结点个数
# - num_layers: LSTM层个数
# - learning_rate: 学习率
# - keep_prob: dropout层中保留结点比例
batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5

epochs = 20
save_every_n = 200
model = CharRNN(len(vocab), batch_size=batch_size,num_steps=num_steps,
                lstm_size=lstm_size,num_layers=num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    counter = 0
    for e in range(epochs):
        # 训练神经网络
        new_state = sess.run(model.initial_state)
        loss = 0
        for x,y in get_batches(encoded,batch_size,num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs:x,
                    model.targets:y,
                    model.keep_prob:keep_prob,
                    model.initial_state:new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                model.final_state,
                                                model.optimizer],
                                                feed_dict=feed)
            end_time = time.time()
            if counter % 100 == 0:
                print('轮数： {} / {}...'.format(e+1, epochs),
                      '训练步数： {}...'.format(counter),
                      '训练误差： {:.4f}...'.format(batch_loss),
                      '{:.4f}： sec /batch'.format((end-start)))
            
            if(counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_1{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_1{}.ckpt".format(counter, lstm_size))
            

# # 5 文本生成
# 现在我们可以基于我们的训练参数进行文本的生成。当我们输入一个字符时，LSTM会预测下一个字符，
# 我们再将新的字符进行输入，这样能不断的循环下去生成本文。
# 
# 为了减少噪音，每次的预测值我会选择最可能的前5个进行随机选择，比如输入h，
# 预测结果概率最大的前五个为[o,e,i,u,b]，我们将随机从这五个中挑选一个作为新的字符，
# 让过程加入随机因素会减少一些噪音的生成。

# In[18]:

def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符
    
    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


# In[19]:

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    """
    生成新文本
    
    checkpoint: 某一轮迭代的参数文件
    n_sample: 新闻本的字符长度
    lstm_size: 隐层结点数
    vocab_size
    prime: 起始文本
    """
    # 将输入的单词转换为单个字符组成的list
    samples = [c for c in prime]
    # sampling=True意味着batch的size=1 x 1
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        # 添加字符到samples中
        samples.append(int_to_vocab[c])
        
        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)


# Here, pass in the path to a checkpoint and sample from the network.

# In[20]:

tf.train.latest_checkpoint('checkpoints')


# In[26]:

# 选用最终的训练参数作为输入进行文本生成
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="The")
print(samp)


# In[22]:

checkpoint = 'checkpoints/i200_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)


# In[23]:

checkpoint = 'checkpoints/i1000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)


# In[24]:

checkpoint = 'checkpoints/i2000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)
