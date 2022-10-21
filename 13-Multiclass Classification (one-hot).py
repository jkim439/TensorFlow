# Logistic Regression: Multiclass Classification (one-hot)

import tensorflow as tf
import numpy as np

xy = np.loadtxt("data/4.csv", delimiter=",", dtype=np.float32)
x_data = xy[:, :-1]
y_data = xy[:, [-1]]
nb_classes = 7

w = tf.Variable(tf.random_normal([16, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

x = tf.placeholder(tf.float32, shape=[None, 16])
y = tf.placeholder(tf.int32, shape=[None, 1])
y_one_hot = tf.reshape(tf.one_hot(y, nb_classes), [-1, nb_classes])

# h = exp(yi) / sigma exp(yi) = softmax
logits = tf.matmul(x, w) + b
h = tf.nn.softmax(logits)

# cost = sigma y*log(h) = cross-entropy
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_one_hot)
)

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# predicted, is_correct, accuracy
predicted = tf.argmax(h, 1)
is_correct = tf.equal(predicted, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_v, acc, _ = sess.run(
            [cost, accuracy, optimizer], feed_dict={x: x_data, y: y_data}
        )
        if step % 500 == 0:
            print("step: {:5}\tcost: {:.3f}\tacc: {:.2%}".format(step, cost_v, acc))

    # Test
    for p, y in zip(sess.run(predicted, feed_dict={x: x_data}), y_data.flatten()):
        print("[{}]\tpred: {}\tlabel: {}".format(p == int(y), p, int(y)))
