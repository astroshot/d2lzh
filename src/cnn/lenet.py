# coding=utf-8
"""
"""
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # 把标签变成one-hot编码的形式
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    n_classes = 10
    # mue = 0
    # sigma = 0.1

    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=n_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    epochs = 10
    batch_size = 128
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss: {}, accuracy: {}'.format(loss, accuracy))
