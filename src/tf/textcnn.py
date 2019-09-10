# coding=utf-8
"""
textCNN
"""

import tensorflow as tf
from keras.layers import (
    Embedding, Conv1D, Input, MaxPooling1D, Concatenate)


class TextCNN(object):

    def __init__(self, vocab, input_length):
        main_input = Input(shape=(50,), dtype='float64')
        embed = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
        embed = embed(main_input)

        cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=48)(cnn1)
        cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=47)(cnn2)
        cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = MaxPooling1D(pool_size=46)(cnn3)

        cnn = Concatenate()

