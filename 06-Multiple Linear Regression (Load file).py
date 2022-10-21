# Multiple Linear Regression (Load file)

import tensorflow as tf
import numpy as np

# Load file using NumPy loadtxt
xy = np.loadtxt("data/1.csv", delimiter=",", dtype=np.float32)
x_data = xy[:, :-1]
y_data = xy[:, [-1]]

print(x_data.shape, len(x_data), x_data)
print(y_data.shape, len(y_data), y_data)

# h = x1w1 + x2w2+ x3w3 + b
w = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
h = tf.matmul(x, w) + b

# cost = 1/m sigma (h-y)^2
cost = tf.reduce_mean(tf.square(h - y))

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7).minimize(cost)

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(4001):
        cost_v, h_v, _ = sess.run(
            [cost, h, optimizer], feed_dict={x: x_data, y: y_data}
        )

    print("My score will be ", sess.run(h, feed_dict={x: [[70, 73, 78]]}))
