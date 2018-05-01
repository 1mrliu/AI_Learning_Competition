import gc
import os
import nltk
import tqdm
import numpy as np
import pandas as pd
nltk.download("punkt")

def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []
    f = open(file_path)

    for index, line in enumerate(f):
        if index == 0:
            continue
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        embedding_list.append(coefs)
        embedding_word_dict[word] = len(embedding_word_dict)
    f.close()
    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.engine import Layer
from keras.layers import Activation, Add, Bidirectional, Conv1D, Dense, Dropout, Embedding, Flatten
from keras.layers import concatenate, GRU, Input, K, LSTM, MaxPooling1D
from keras.layers import GlobalAveragePooling1D,  GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.3
rate_drop_dense = 0.3

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
        input1 = Input (shape=(sequence_length,))
        embed_layer = Embedding (embedding_matrix.shape[0], embedding_matrix.shape[1],
                                 weights=[embedding_matrix], trainable=False) (input1)
        embed_layer = SpatialDropout1D (rate_drop_dense) (embed_layer)

        x = Bidirectional (
            GRU (gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True)) (
            embed_layer)
        capsule = Capsule (num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                           share_weights=True) (x)
        capsule = Flatten () (capsule)
        capsule = Dropout (dropout_p) (capsule)
        output = Dense (1, activation='sigmoid') (capsule)
        model = Model (inputs=input1, outputs=output)
        model.compile (
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        return model


def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    num_labels = train_y.shape[1]
    patience = 5
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    while True:
        model.fit (train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict (val_x, batch_size=batch_size)

        total_loss = 0
        for j in range (num_labels):
            loss = log_loss (val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= num_labels

        print ("Epoch {0} loss {1} best_loss {2}".format (current_epoch, total_loss, best_loss))

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights ()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == patience:
                break

    model.set_weights (best_weights)
    return model

def train_folds(X, y, X_test, fold_count, batch_size, get_model_func):
    print("="*75)
    fold_size = len(X) // fold_count
    models = []
    result_path = "predictions"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = np.array(X[fold_start:fold_end])
        val_y = np.array(y[fold_start:fold_end])

        model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y)
        train_predicts_path = os.path.join(result_path, "train_predicts{0}.npy".format(fold_id))
        test_predicts_path = os.path.join(result_path, "test_predicts{0}.npy".format(fold_id))
        train_predicts = model.predict(X, batch_size=512, verbose=1)
        test_predicts = model.predict(X_test, batch_size=512, verbose=1)
        np.save(train_predicts_path, train_predicts)
        np.save(test_predicts_path, test_predicts)

    return models



# 加载训练集
train_data = pd.read_csv('/Users/liudong/Desktop/DonorsChoose/train.csv')
print(train_data)

# 加载测试集
test_data  = pd.read_csv('/Users/liudong/Desktop/DonorsChoose/test.csv')
print(test_data)

# 加载资源集
resource_data = pd.read_csv('/Users/liudong/Desktop/DonorsChoose/resources.csv')
print(resource_data)