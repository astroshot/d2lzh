# coding=utf-8
"""
"""
import keras
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.layers import (
    Input, Embedding, Lambda, Conv2D, MaxPooling2D, Flatten,
    Dropout, Dense, Concatenate
)
from keras.models import Model
from matplotlib import pyplot as plt


def load_imdb():
    """To fix ValueError: Object arrays cannot be loaded when allow_pickle=False"""
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    # call load_data with allow_pickle implicitly set to true
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # restore np.load for future normal usage
    np.load = np_load_old
    return (train_data, train_labels), (test_data, test_labels)


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def _reshapes(embed):
    return tf.reshape(embed, (-1, 256, 32, 1))


if __name__ == '__main__':
    """
    参数与超参数：

    - sequence_length （Q: 对于CNN, 输入与输出都是固定的，可每个句子长短不一, 怎么处理? 
        A: 需要做定长处理, 比如定为n, 超过的截断, 不足的补0. 注意补充的0对后面的结果没有影响，
        因为后面的max-pooling只会输出最大值，补零的项会被过滤掉）
    - num_classes （多分类, 分为几类）
    - vocabulary_size （语料库的词典大小, 记为|D|）
    - embedding_size （将词向量的维度, 由原始的 |D| 降维到 embedding_size）
    - filter_size_arr （多个不同size的filter）
    
    2015年“A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification”论文详细地阐述了关于TextCNN模型的调参心得
    --------------------- 

    - 使用预训练的word2vec 、 GloVe初始化效果会更好。一般不直接使用One-hot。
    - 卷积核的大小影响较大，一般取1~10，对于句子较长的文本，则应选择大一些。
    - 卷积核的数量也有较大的影响，一般取100~600 ，同时一般使用Dropout（0~0.5）。
    - 激活函数一般选用ReLU 和 tanh。
    - 池化使用1-max pooling。
    - 随着feature map数量增加，性能减少时，试着尝试大于0.5的Dropout。
    - 评估模型性能时，记得使用交叉验证。
    """
    (train_data, train_labels), (test_data, test_labels) = load_imdb()

    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    print(decode_review(train_data[0]))

    train_data = keras.preprocessing.sequence.pad_sequences(
        train_data, value=word_index['<PAD>'], padding='post', maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(
        test_data, value=word_index['<PAD>'], padding='post', maxlen=256)

    convs = []
    inputs = Input(shape=(256,))
    embed1 = Embedding(10000, 32)(inputs)
    embed = Lambda(_reshapes)(embed1)

    conv1 = Conv2D(filters=3, kernel_size=(2, 32), activation='relu')(embed)
    pool1 = MaxPooling2D(pool_size=(255, 1))(conv1)
    flatten_pool1 = Flatten()(pool1)
    convs.append(flatten_pool1)

    conv2 = Conv2D(filters=3, kernel_size=(3, 32), activation='relu')(embed)
    pool2 = MaxPooling2D(pool_size=(254, 1))(conv2)
    flatten_pool2 = Flatten()(pool2)
    convs.append(flatten_pool2)

    conv3 = Conv2D(filters=3, kernel_size=(4, 32), activation='relu')(embed)
    pool3 = MaxPooling2D(pool_size=(253, 1))(conv3)
    flatten_pool3 = Flatten()(pool3)
    convs.append(flatten_pool3)

    merge = Concatenate(axis=1)(convs)
    out = Dropout(0.5)(merge)
    output = Dense(units=32, activation='relu')(out)
    pred = Dense(units=1, activation='sigmoid')(output)

    model = Model(inputs=inputs, outputs=pred)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    x_val = train_data[:10000]
    partial_x_val = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_val = train_labels[10000:]

    history = model.fit(partial_x_val, partial_y_val, batch_size=512, epochs=40, validation_data=(x_val, y_val))
    result = model.evaluate(test_data, test_labels)
    print(result)

    history_dict = history.history
    print(history_dict.keys())

    predictions = model.predict(test_data)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    # # -------------画图方式1-----------------

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    # -----------------------------------------
    plt.clf()  # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
