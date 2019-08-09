# coding=utf-8
"""
"""
import os
import random
import urllib.request
import zipfile
import math
from collections import Counter, deque

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

data_index = 0


def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(cwd, '../data')
    filename = os.path.join(data_path, filename)

    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    counter = Counter(words)
    count.extend(counter.most_common(n_words - 1))
    word2idx = {}
    for word in count:
        word2idx[word[0]] = len(word2idx)
    # for index, word in enumerate(count):
    #     word2idx[word] = index

    data = []
    unk_count = 0

    for word in words:
        if word in word2idx:
            index = word2idx[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count  # 'UNK' to unk_count
    index2word = dict(zip(word2idx.values(), word2idx.keys()))
    return data, count, word2idx, index2word


def collect_data(vocab_size=10000):
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', url, 31344016)
    vocab = read_data(filename)
    print(vocab[:7])
    data, count, word2idx, index2word = build_dataset(vocab, vocab_size)
    return data, count, word2idx, index2word


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            context[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    data_index = (data_index + len(data) - span) % len(data)
    return batch, context


class SimilarityCallback(object):

    def __init__(self, idx2word):
        self.idx2word = idx2word

    def run_sim(self):
        for i in range(valid_size):
            valid_word = self.idx2word[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.idx2word[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim


if __name__ == '__main__':
    vocab_size = 10000
    data, count, word2idx, index2word = collect_data(vocab_size)

    window_size = 3
    vector_dim = 300
    epochs = 20000

    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    sampling_table = sequence.make_sampling_table(vocab_size)
    couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype='int32')
    word_context = np.array(word_context, dtype='int32')
    print(couples[:10], labels[:10])

    input_target = Input((1,))
    input_context = Input((1,))

    embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
    target = embedding(input_target)
    target = Reshape((vector_dim, 1))(target)
    context = embedding(input_context)
    context = Reshape((vector_dim, 1))(context)

    similarity = merge([target, context], mode='cos', dot_axes=0)
    # now perform the dot product operation to get a similarity measure
    dot_product = merge([target, context], mode='dot', dot_axes=1)
    dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)
    # create the primary training model
    model = Model(input=[input_target, input_context], output=output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # create a secondary validation model to run our similarity checks during training
    validation_model = Model(input=[input_target, input_context], output=similarity)

    sim_cb = SimilarityCallback(idx2word=index2word)

    arr_1 = np.zeros((1,))
    arr_2 = np.zeros((1,))
    arr_3 = np.zeros((1,))
    for cnt in range(epochs):
        idx = np.random.randint(0, len(labels) - 1)
        arr_1[0,] = word_target[idx]
        arr_2[0,] = word_context[idx]
        arr_3[0,] = labels[idx]
        loss = model.train_on_batch([arr_1, arr_2], arr_3)
        if cnt % 100 == 0:
            print("Iteration {}, loss={}".format(cnt, loss))
        if cnt % 10000 == 0:
            sim_cb.run_sim()

    # batch_size = 128
    # embedding_size = 128
    # skip_window = 1
    # num_skips = 2
    #
    # inputs = tf.placeholder(tf.int32, shape=(batch_size,))
    # outputs = tf.placeholder(tf.int32, shape=(batch_size, 1))
    # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    #
    # embedding = tf.Variable(tf.random_uniform((vocab_size, embedding_size), -1, 1))
    # embed = tf.nn.embedding_lookup(embedding, inputs)
    #
    # weights = tf.Variable(tf.truncated_normal((vocab_size, embedding_size), stddev=1.0 / math.sqrt(embedding_size)))
    # biases = tf.Variable(tf.zeros(vocab_size))
    # hidden_out = tf.matmul(embed, tf.transpose(weights), ) + biases
    #
    # batch, train_context = generate_batch(vocab, batch_size, num_skips, skip_window)
    #
    # train_one_hot = tf.one_hot(train_context, vocab_size)
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))
    # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)
    #
    # norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    # normalized_embedding = embedding / norm
    # valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # similarity = tf.matmul(valid_embedding, normalized_embedding, transpose_b=True)
