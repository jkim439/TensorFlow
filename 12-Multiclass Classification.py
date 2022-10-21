# Logistic Regression: Multiclass Classification

import tensorflow as tf

x_data = [
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7],
]
y_data = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
]

w = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable(tf.random_normal([3]))
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

# h = exp(yi) / sigma exp(yi) = softmax
logits = tf.matmul(x, w) + b
h = tf.nn.softmax(logits)

# cost = -sigma y*log(h)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})

    a = sess.run(h, feed_dict={x: [[1, 2, 5, 6]]})
    print(a, sess.run(tf.argmax(a, 1)))
