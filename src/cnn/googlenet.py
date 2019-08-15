# coding=utf-8
"""
"""
import os

from keras.models import Model
from keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate
)
import numpy as np

seed = 7
np.random.seed(seed)


def conv2d_bn(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + 'bn'
        conv_name = name + 'conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def inception(x, nb_filter):
    branch_1x1 = conv2d_bn(x, nb_filter, (1, 1))

    branch_3x3 = conv2d_bn(x, nb_filter, (1, 1))
    branch_3x3 = conv2d_bn(branch_3x3, nb_filter, (3, 3))

    branch_5x5 = conv2d_bn(x, nb_filter, (1, 1))
    branch_5x5 = conv2d_bn(branch_5x5, nb_filter, (5, 5))

    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, nb_filter, (1, 1))

    x = concatenate([branch_1x1, branch_3x3, branch_5x5, branch_pool], axis=3)
    return x


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    input_layer = Input(shape=input_shape)
    x = conv2d_bn(input_layer, 64, (7, 7), strides=(2, 2))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = conv2d_bn(x, 192, (3, 3), strides=(1, 1))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception(x, 64)
    x = inception(x, 120)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception(x, 128)
    x = inception(x, 128)
    x = inception(x, 128)
    x = inception(x, 132)
    x = inception(x, 208)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception(x, 208)
    x = inception(x, 256)
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(1000, activation='softmax')(x)

    model = Model(input_layer, name='inception')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.build(input_shape)
    model.summary()
