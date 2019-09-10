# coding=utf-8
"""
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

if __name__ == '__main__':
    num = 100

    learning_rate = 0.05
    learning_epochs = 5000
    display_step = 50

    train_X = np.asarray(
        [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray(
        [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    n_samples = train_X.shape[0]

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    W = tf.Variable(np.random.randn(), name='weight')
    b = tf.Variable(np.random.randn(), name='bias')

    pred = tf.add(tf.multiply(X, W), b)
    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / 2 / n_samples
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        for epoch in range(learning_epochs):
            for x, y in zip(train_X, train_Y):
                session.run(optimizer, feed_dict={X: x, Y: y})

            if epoch % display_step == 0:
                c = session.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                      "W=", session.run(W), "b=", session.run(b))

        print('Optimization finished')
        training_cost = session.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, "W=", session.run(W), "b=", session.run(b), '\n')

        plt.plot(train_X, train_Y, 'ro', label='Original data')
        plt.plot(train_X, session.run(W) * train_X + session.run(b), label='Fitted line')
        plt.legend()
        plt.show()
