# coding=utf-8
"""RNN Model implemented by tensorflow
"""

import tensorflow as tf


class RNN(object):
    """RNN Model implemented by tensorflow"""

    def __init__(self, num_input, num_hidden, num_output, input_data):
        self.idx_to_char = {}
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.params = None
        self.state = None

        self.input_data = input_data

    def get_params(self):

        # hidden layer
        w_xh = tf.Variable(tf.random_normal((self.num_input, self.num_hidden)))
        w_hh = tf.Variable(tf.random_normal(shape=(self.num_hidden, self.num_hidden)))
        b_h = tf.Variable(tf.zeros((self.num_input, self.num_hidden)))

        # output layer
        w_hq = tf.Variable(tf.random_normal((self.num_hidden, self.num_output)))
        b_q = tf.Variable(tf.zeros((self.num_input, )))

        params = (w_xh, w_hh, b_h, w_hq, b_q)
        self.params = params

    def rnn(self):
        w_xh, w_hh, b_h, w_hq, b_q = self.params
        H = self.state
        outputs = []

        for x in self.input_data:
            sigma = tf.multiply(x, w_xh) + tf.multiply(H, w_hh) + b_h
            H = tf.tanh(sigma)
            Y = tf.multiply(H, w_hq) + b_q
            outputs.append(Y)

        return outputs, H
