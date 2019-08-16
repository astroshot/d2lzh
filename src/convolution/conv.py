# coding=utf-8
"""
"""
import tensorflow as tf

if __name__ == '__main__':
    input_mat = tf.constant([[1, 1, 1, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 1, 1, 1],
                             [0, 0, 1, 1, 0],
                             [0, 1, 1, 0, 0]], shape=(1, 5, 5, 1), dtype=tf.float32)
    # 卷积核矩阵
    filter_mat = tf.constant([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]], shape=(3, 3, 1, 1), dtype=tf.float32)

    op1 = tf.nn.conv2d(input_mat, filter_mat, strides=[1, 1, 1, 1], padding='VALID')
    # 卷积计算
    op2 = tf.nn.conv2d(input_mat, filter_mat, strides=[1, 1, 1, 1], padding='SAME')
    with tf.Session() as sess:
        result1 = sess.run(op1)
        result2 = sess.run(op2)
        print(result1)
        print('###############')
        print(result2)
