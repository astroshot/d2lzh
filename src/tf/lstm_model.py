# coding=utf-8
"""
"""
import pickle
import os
import tensorflow as tf
import jieba
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

_DIR = os.path.dirname(os.path.abspath(__file__))


class LSTMModel:

    def __init__(self, input_data, output_targets, num_words, num_units, num_layers, batch_size):
        self.input_data = input_data
        self.output_targets = output_targets
        self.num_words = num_words
        self.num_units = num_units
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.tensors = None

    def build(self):
        self.tensors = {}

        with tf.name_scope('embedding'):
            w = tf.Variable(tf.random_uniform((self.num_words, self.num_units)), -1.0, 1.0, name='W')
            inputs = tf.nn.embedding_lookup(w, self.input_data)

        with tf.name_scope('lstm'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
            cells = [lstm_cell(self.num_units, state_is_tuple=True) for i in range(self.num_layers)]
            cell_mul = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

            initial_state = cell_mul.zero_state(self.batch_size, dtype=tf.float32)
            outputs, last_state = tf.nn.dynamic_rnn(cell_mul, inputs, initial_state=initial_state)

        with tf.name_scope('softmax'):
            output = tf.reshape(outputs, [-1, self.num_units])
            weights = tf.Variable(tf.truncated_normal([self.num_units, self.num_words]))
            bias = tf.Variable(tf.zeros(shape=[self.num_words]))
            logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)

        if self.batch_size > 1:
            with tf.name_scope('loss'):
                labels = tf.reshape(self.output_targets, (-1,))
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                total_loss = tf.reduce_mean(loss)

            train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
            self.tensors['initial_state'] = initial_state
            self.tensors['output'] = output
            self.tensors['train_op'] = train_op
            self.tensors['total_loss'] = total_loss
            self.tensors['loss'] = loss
            self.tensors['last_state'] = last_state

        else:
            prediction = tf.nn.softmax(logits)
            self.tensors['initial_state'] = initial_state
            self.tensors['last_state'] = last_state
            self.tensors['prediction'] = prediction


class DataProcess:

    def __init__(self, texts=None, len_max=None, len_min=None, word_index=None, num_words=None,
                 texts_seq=None, x_pad_seq=None, y_pad_seq=None, y_one_hot=None, maxlen=None):
        self.texts = texts
        self.len_max = len_max
        self.len_min = len_min
        self.word_index = word_index
        self.num_words = num_words
        self.texts_seq = texts_seq
        self.x_pad_seq = x_pad_seq
        self.y_pad_seq = y_pad_seq
        self.y_one_hot = y_one_hot
        self.maxlen = maxlen
        self.file_type = None

    def load_data(self, file_type='poem', len_min=0, len_max=200):
        if file_type == 'poem':
            with open(_DIR + '/data/Tang_poem.pkl', mode='rb') as f:
                texts = pickle.load(f)
            texts = [i for i in texts if len_min <= len(i) <= len_max]
        elif file_type == 'story':
            with open(_DIR + '/data/story.pkl', mode='rb') as f:
                texts = pickle.load(f)
            texts = [''.join(texts), 'E']
        else:
            raise ValueError('file_type should be poem or story')

        self.texts = texts
        self.len_min = len_min
        self.len_max = len_max
        self.file_type = file_type

    def text2seq(self, mode='length', num_words=None, maxlen=40, cut=False):
        self.mode = mode
        texts = self.texts

        if self.file_type == 'poem':
            cut = False
        if cut:
            texts = [jieba.lcut(text) for text in texts]
            print('cut finished')
            tokenizer = Tokenizer(num_words=num_words, char_level=False)
        else:
            tokenizer = Tokenizer(num_words=num_words, char_level=True)

        if self.mode == 'sample':
            pass
        elif self.mode == 'length':
            new_texts = []
            for i in texts:
                mod = len(i) * maxlen
                i += ('E' * (maxlen - mod))
                for j in range(len(i) // maxlen + 1):
                    new_texts.append(i[j * maxlen: (j * maxlen + maxlen)])
            texts = new_texts
            self.maxlen = maxlen
        else:
            raise ValueError('file_type should be poem or story')

        tokenizer.fit_on_texts(texts)
        word_index = tokenizer.word_index
        self.word_index = word_index
        num_words = min(num_words, len(word_index.keys()) + 1)
        self.num_words = num_words

        texts_seq = tokenizer.texts_to_sequences(texts)
        self.texts_seq = texts_seq

    def create_one_hot(self, y, num_words):
        y_one_hot = np.zeros(shape=(len(y), num_words))
        for num, i in enumerate(y):
            y_one_hot[num, i] = 1
        return y_one_hot

    def create_x_y(self, maxlen=40, one_hot=False):
        """

        :param maxlen: max length
        :param one_hot: 是否转 one hot
        :return:
        """
        self.one_hot = one_hot
        if self.maxlen is not None:
            maxlen = self.maxlen
        texts_seq = self.texts_seq
        x = []
        y = []
        for i in texts_seq:
            x.append(i[:-1])
            y.append(i[1:])

        n = 0
        pad_seq = []

        while n < len(texts_seq):
            pad_seq += list(pad_sequences(x[n:n + 5000], maxlen=maxlen, padding='post', value=0, dtype=tf.int32))
            n += 5000

        pad_seq = pad_sequences(x, maxlen, padding='post', truncating='post')
        pad_seq_y = pad_sequences(y, maxlen - 1, padding='post', truncating='post')

        self.x_pad_seq = np.array([i[:-1] for i in pad_seq])
        self.y_pad_seq = np.array([i[1:] for i in pad_seq])

        if one_hot:
            y_one_hot = [self.create_one_hot(i, self.num_words) for i in self.y_pad_seq]
            self.y_one_hot = y_one_hot

    def transform(self,
                  num_words=6000,
                  mode='sample',
                  len_min=0,
                  len_max=5,
                  maxlen=40,
                  one_hot=False,
                  file_type='poem',
                  cut=False):
        self.load_data(file_type=file_type, len_min=len_min, len_max=len_max)
        self.texts_seq(mode=mode, num_words=num_words, maxlen=maxlen, cut=cut)
        self.create_x_y(maxlen=maxlen, one_hot=one_hot)
        x= np.array(self.x_pad_seq)

        if one_hot:
            y = np.array(self.y_one_hot)
        else:
            y = np.array(self.y_pad_seq)
        return x, y, self.word_index


def train(maxlen=40,
          batchsize=64,
          num_words=4000,
          num_units=128,
          num_layers=2,
          epochs=1,
          mode='length',
          file_type='poem',
          len_min=0,
          len_max=10e8,
          cut=False,
          process_path=_DIR + '/model/poem/data_process.pkl',
          model_path=_DIR + '/model/poem/train'):

    processer = DataProcess()
    x, y, word_index = processer.transform(num_words=num_words, mode=mode, len_min=len_min, len_max=len_max,
                                           maxlen=maxlen, file_type=file_type, cut=cut)
    processer.num_units = num_units
    processer.num_layers = num_layers

    with open(process_path, 'rb') as f:
        pickle.dump(processer, f)


