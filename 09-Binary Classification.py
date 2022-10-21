# Logistic Regression: Binary Classification

import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

w = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# h = 1 / 1 + exp(-xw) = sigmoid
h = tf.sigmoid(tf.matmul(x, w) + b)

# cost = 1/m sigma -ylog(h) - (1-y)log(1-h)
cost = -tf.reduce_mean(y * tf.log(h) + (1 - y) * tf.log(1 - h))

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# predicted, accuracy
predicted = tf.cast(h > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_v, _ = sess.run([cost, optimizer], feed_dict={x: x_data, y: y_data})

    # Result
    hv, pv, av = sess.run([h, predicted, accuracy], feed_dict={x: x_data, y: y_data})
    print("\nHypothesis: ", hv, "\nPredicted: ", pv, "\nAccuracy: ", av)

    # Predict
    print("My result will be ", sess.run(h, feed_dict={x: [[4, 3]]}))  # 0.6732117
