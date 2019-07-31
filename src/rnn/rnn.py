# coding=utf-8
"""
"""

import d2lzh as d2l
import os
import math
import zipfile
import time

from mxnet import autograd, nd, context
from mxnet.gluon import loss as gloss


class RNNTestModel(object):

    def __init__(self, num_epochs, num_steps, batch_size, lr, clipping_theta):
        (corpus_indices, char_to_idx, idx_to_char, vocab_size) = self.load_data_jay_lyrics()

        self.corpus_indices = corpus_indices
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size

        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.lr = lr
        self.clipping_theta = clipping_theta

        # self.ctx = context.cpu()
        self.ctx = d2l.try_gpu()
        print('using {}'.format(self.ctx))

        self.num_hidden = 256  # 隐藏层数目
        self.num_input = self.vocab_size
        self.num_output = self.vocab_size

        self.params = None
        self.state = None
        self.inputs = None
        self.outputs = None

    @staticmethod
    def to_onehot(X, size):
        return [nd.one_hot(x, size) for x in X.T]

    @staticmethod
    def load_data_jay_lyrics():
        """Load the Jay Chou lyric data set (available in the Chinese book)."""
        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(cwd, '../data')
        lyrics_file = os.path.join(data_path, 'jaychou_lyrics.txt.zip')

        with zipfile.ZipFile(lyrics_file) as zin:
            with zin.open('jaychou_lyrics.txt') as f:
                corpus_chars = f.read().decode('utf-8')
        corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
        corpus_chars = corpus_chars[0:10000]
        idx_to_char = list(set(corpus_chars))
        char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
        vocab_size = len(char_to_idx)
        corpus_indices = [char_to_idx[char] for char in corpus_chars]
        return corpus_indices, char_to_idx, idx_to_char, vocab_size

    def get_params(self):
        def _one(shape):
            return nd.random.normal(scale=0.1, shape=shape, ctx=self.ctx)

        # 隐藏层参数
        W_xh = _one((self.num_input, self.num_hidden))
        W_hh = _one((self.num_hidden, self.num_hidden))
        b_h = nd.zeros(self.num_hidden, ctx=self.ctx)

        # 输出层参数
        W_hq = _one((self.num_hidden, self.num_output))
        b_q = nd.zeros(self.num_output, ctx=self.ctx)

        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.attach_grad()
        self.params = params

    def init_rnn_state(self, batch_size, num_hidden):
        self.state = nd.zeros(shape=(batch_size, num_hidden), ctx=self.ctx),

    def rnn(self):
        W_xh, W_hh, b_h, W_hq, b_q = self.params
        H, = self.state
        outputs = []

        for X in self.inputs:
            H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
            Y = nd.dot(H, W_hq) + b_q
            outputs.append(Y)

        return outputs, (H,)

    def predict(self, prefix, num_chars):
        self.init_rnn_state(1, self.num_hidden)
        output = [self.char_to_idx[prefix[0]]]
        for t in range(num_chars + len(prefix) - 1):
            self.inputs = self.to_onehot(nd.array([output[-1]], ctx=self.ctx), self.vocab_size)
            Y, state = self.rnn()

            if t < len(prefix) - 1:
                output.append(self.char_to_idx[prefix[t + 1]])
            else:
                output.append(int(Y[0].argmax(axis=1).asscalar()))
        return ''.join([self.idx_to_char[i] for i in output])

    def grad_clipping(self, theta):
        norm = nd.array([0], self.ctx)
        for param in self.params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in self.params:
                param.grad[:] *= theta / norm

    def train_and_predict(self, is_random_iter, pred_period, pred_len, prefixes):
        if is_random_iter:
            data_iter_fn = d2l.data_iter_random
        else:
            data_iter_fn = d2l.data_iter_consecutive

        self.get_params()
        loss = gloss.SoftmaxCrossEntropyLoss()

        for epoch in range(self.num_epochs):
            if not is_random_iter:
                self.init_rnn_state()
            l_sum, n, start = 0.0, 0, time.time()
            data_iter = data_iter_fn(self.corpus_indices, self.batch_size, self.num_steps)
            for X, Y in data_iter:
                if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                    self.init_rnn_state(self.batch_size, self.num_hidden)
                else:  # 否则需要使用detach函数从计算图分离隐藏状态
                    for s in self.state:
                        s.detach()
                with autograd.record():
                    self.inputs = self.to_onehot(X.as_in_context(self.ctx), self.vocab_size)
                    outputs, state = self.rnn()
                    outputs = nd.concat(*outputs, dim=0)
                    y = Y.T.reshape((-1,))
                    l = loss(outputs, y).mean()
                l.backward()
                self.grad_clipping(self.clipping_theta)
                d2l.sgd(self.params, self.lr, 1)
                l_sum += l_sum
                n += y.size

            if (epoch + 1) % pred_period == 0:
                print(
                    'epoch {}, perplexity {}, time {} sec'.format(epoch + 1, math.exp(l_sum / n), time.time() - start))
                for prefix in prefixes:
                    print(' -', self.predict(prefix, pred_len))


if __name__ == '__main__':
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    rnn = RNNTestModel(num_epochs=256, num_steps=35, batch_size=32, lr=1e2, clipping_theta=1e-2)
    rnn.train_and_predict(True, pred_period, pred_len, prefixes)
