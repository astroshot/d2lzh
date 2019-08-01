# coding=utf-8
"""
"""
import math
import time

import d2lzh as d2l
from mxnet import nd, context, autograd
from mxnet.gluon import loss as gloss

from src.rnn.util import load_data_jay_lyrics, to_onehot, grad_clipping


class LSTMTest(object):

    def __init__(self, batch_size, num_steps):
        corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()

        self.corpus_indices = corpus_indices
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size

        self.num_inputs = vocab_size
        self.num_hiddens = 256
        self.num_outputs = vocab_size

        self.batch_size = batch_size
        self.num_steps = num_steps

        self.ctx = d2l.try_gpu()
        self.params = None

    def get_params(self):
        def _one(shape):
            return nd.random.normal(scale=0.01, shape=shape, ctx=self.ctx)

        def _three():
            return (_one((self.num_inputs, self.num_hiddens)), _one((self.num_hiddens, self.num_hiddens)),
                    nd.zeros(self.num_hiddens, ctx=self.ctx))

        W_xi, W_hi, b_i = _three()
        W_xf, W_hf, b_f = _three()
        W_xo, W_ho, b_o = _three()
        W_xc, W_hc, b_c = _three()

        W_hq = _one((self.num_hiddens, self.num_outputs))
        b_q = nd.zeros(self.num_outputs, ctx=self.ctx)

        params = [
            W_xi, W_hi, b_i,
            W_xf, W_hf, b_f,
            W_xo, W_ho, b_o,
            W_xc, W_hc, b_c,
            W_hq, b_q
        ]

        for param in params:
            param.attach_grad()

        self.params = params

    def init_lstm_state(self, batch_size):
        return (nd.zeros(shape=(batch_size, self.num_hiddens), ctx=self.ctx),
                nd.zeros(shape=(batch_size, self.num_hiddens), ctx=self.ctx))

    def lstm(self, inputs, state):
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
         W_hq, b_q] = self.params
        (H, C) = state
        outputs = []

        for X in inputs:
            I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
            F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
            O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
            C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)

            C = I * C_tilda + F * C
            H = nd.tanh(C) * O
            Y = nd.dot(H, W_hq) + b_q
            outputs.append(Y)

        return outputs, (H, C)

    def predict(self, prefix, num_chars):
        """Predict next chars with a RNN model"""
        state = self.init_lstm_state(1)
        output = [self.char_to_idx[prefix[0]]]
        for t in range(num_chars + len(prefix) - 1):
            X = to_onehot(nd.array([output[-1]], ctx=self.ctx), self.vocab_size)
            (Y, state) = self.lstm(X, state)
            if t < len(prefix) - 1:
                output.append(self.char_to_idx[prefix[t + 1]])
            else:
                output.append(int(Y[0].argmax(axis=1).asscalar()))
        return ''.join([self.idx_to_char[i] for i in output])

    def train_and_predict(self, is_random_iter, num_epochs, lr, clipping_theta, pred_period, pred_len, prefixes):
        if is_random_iter:
            data_iter_fn = d2l.data_iter_random
        else:
            data_iter_fn = d2l.data_iter_consecutive
        self.get_params()
        loss = gloss.SoftmaxCrossEntropyLoss()

        for epoch in range(num_epochs):
            if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
                state = self.init_lstm_state(self.batch_size)
            l_sum, n, start = 0.0, 0, time.time()
            data_iter = data_iter_fn(self.corpus_indices, self.batch_size, self.num_steps, self.ctx)
            for X, Y in data_iter:
                if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                    state = self.init_lstm_state(self.batch_size)
                else:  # 否则需要使用detach函数从计算图分离隐藏状态
                    for s in state:
                        s.detach()
                with autograd.record():
                    inputs = to_onehot(X, self.vocab_size)
                    # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                    (outputs, state) = self.lstm(inputs, state)
                    # 拼接之后形状为(num_steps * batch_size, vocab_size)
                    outputs = nd.concat(*outputs, dim=0)
                    # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                    # batch * num_steps 的向量，这样跟输出的行一一对应
                    y = Y.T.reshape((-1,))
                    # 使用交叉熵损失计算平均分类误差
                    l = loss(outputs, y).mean()
                l.backward()
                grad_clipping(self.params, clipping_theta, self.ctx)  # 裁剪梯度
                d2l.sgd(self.params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
                l_sum += l.asscalar() * y.size
                n += y.size

            if (epoch + 1) % pred_period == 0:
                print('epoch %d, perplexity %f, time %.2f sec' % (
                    epoch + 1, math.exp(l_sum / n), time.time() - start))
                for prefix in prefixes:
                    print(' -', self.predict(prefix, pred_len))


if __name__ == '__main__':
    num_epochs = 160
    num_steps = 35
    batch_size = 32
    lr = 1e2
    clipping_theta = 1e-2
    pred_period = 40
    pred_len = 50
    prefixes = ['分开', '不分开']

    l = LSTMTest(batch_size, num_steps)
    l.train_and_predict(False, num_epochs, lr, clipping_theta, pred_period, pred_len, prefixes)
