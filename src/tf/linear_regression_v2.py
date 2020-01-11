# coding=utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = np.linspace(0, 100, 30)
    y = 3 * x + 7 + np.random.randn(30) * 7

    model = Sequential([Dense(1, input_shape=(1,))])
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    history = model.fit(x, y, epochs=50)
    plt.scatter(x, y)
    plt.plot(x, model.predict(x))
    plt.show()

