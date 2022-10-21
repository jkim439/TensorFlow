# Linear Regression (gvs)

import tensorflow as tf

# h = wx + b
w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
h = w * x + b

# cost = 1/m sigma (h-y)^2
cost = tf.reduce_mean(tf.square(h - y))

# Gradient Descent Optimizer (without minimize)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# gvs
gvs = optimizer.compute_gradients(cost)
apply = optimizer.apply_gradients(gvs)

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        _, cost_v, w_v, b_v = sess.run(
            [apply, cost, w, b], feed_dict={x: [1, 2, 3], y: [2.1, 3.1, 4.1]}
        )
        if step % 1000 == 0:
            print(step, cost_v, w_v, b_v)
