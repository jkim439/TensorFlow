# Logistic Regression: Binary Classification (XOR)

import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

# Layer 1
w1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.random_normal([2]))
x = tf.placeholder(tf.float32, shape=[4, 2])
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# Layer 2
w2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.random_normal([1]))
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

# y
y = tf.placeholder(tf.float32, shape=[4, 1])

# cost = 1/m sigma -ylog(h) - (1-y)log(1-h)
cost = -tf.reduce_mean(y * tf.log(layer2) + (1 - y) * tf.log(1 - layer2))

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# predicted, accuracy
predicted = tf.cast(layer2 > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_v, _ = sess.run([cost, optimizer], feed_dict={x: x_data, y: y_data})
        if step % 1000 == 0:
            print(cost_v)

    # Test
    h, p, a = sess.run(
        [layer2, predicted, accuracy],
        feed_dict={x: x_data, y: y_data},
    )
    print("Hypothesis:\n", h)
    print("Predicted Class\n", p)
    print("Accuracy\n", a)
