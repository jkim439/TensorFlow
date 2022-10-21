# Multiple Linear Regression

import tensorflow as tf

# x1, x2, x3
x_data = [
    [73.0, 80.0, 75.0],
    [93.0, 88.0, 93.0],
    [89.0, 91.0, 90.0],
    [96.0, 98.0, 100.0],
    [73.0, 66.0, 70.0],
]
y_data = [[152.0], [185.0], [180.0], [196.0], [142.0]]

# h = x1w1 + x2w2+ x3w3 + b
w = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
h = tf.matmul(x, w) + b

# cost = 1/m sigma (h-y)^2
cost = tf.reduce_mean(tf.square(h - y))

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_v, h_v, w_v, b_v, _ = sess.run(
            [cost, h, w, b, optimizer], feed_dict={x: x_data, y: y_data}
        )
        if step % 500 == 0:
            print(step, cost_v, h_v, w_v, b_v)

    print(sess.run([h], feed_dict={x: [[89.0, 91.0, 90.0]]}))
