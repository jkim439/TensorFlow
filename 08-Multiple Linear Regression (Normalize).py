# Multiple Linear Regression (Normalize)

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import minmax_scale

xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

# minmax_scale
xy = minmax_scale(xy)

x_data = xy[:, :-1]
y_data = xy[:, [-1]]

# h = x1w1 + x2w2+ x3w3 + x4w4 + b
w = tf.Variable(tf.random_normal([4, 1]))
b = tf.Variable(tf.random_normal([1]))
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 1])
h = tf.matmul(x, w) + b

# cost = 1/m sigma (h-y)^2
cost = tf.reduce_mean(tf.square(h - y))

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _, cost_v, h_v = sess.run(
            [optimizer, cost, h], feed_dict={x: x_data, y: y_data}
        )
        print(step, "Cost: ", cost_v, "\nPrediction:\n", h_v)

    print(
        sess.run([h], feed_dict={x: [[0.49556179, 0.4258239, 0.31521739, 0.48131134]]})
    )  # 0.49276137
