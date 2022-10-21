# Convolutional Neural Network (CNN)

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])  # n x 784
y_one_hot = tf.placeholder(tf.float32, shape=[None, 10])
img = tf.reshape(x, [-1, 28, 28, 1])  # 28 x 28 x 1color x nimg

# CNN Layer 1: CONV - ReLU - POOL
w1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))  # 3 x 3 x 1color x 32filter
l1 = tf.nn.conv2d(img, w1, strides=[1, 1, 1, 1], padding="SAME")  # 28 x 28 x 32
l1 = tf.nn.relu(l1)  # 28 x 28 x 32
l1 = tf.nn.max_pool(
    l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
)  # 14 x 14 x 32

# CNN Layer 2: CONV - ReLU - POOL
w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))  # 3 x 3 x 32color x 64filter
l2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding="SAME")  # 14 x 14 x 64
l2 = tf.nn.relu(l2)  # 14 x 14 x 64
l2 = tf.nn.max_pool(
    l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
)  # 7 x 7 x 64

# Fully Connected (FC), Dense Layer
flat = tf.reshape(l2, [-1, 7 * 7 * 64])  # n x 3136
w3 = tf.get_variable(
    "w3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer()
)
b = tf.Variable(tf.random_normal([10]))

# h = exp(yi) / sigma exp(yi) = softmax
logits = tf.matmul(flat, w3) + b
h = tf.nn.softmax(logits)

# cost = sigma y*log(h) = cross-entropy
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_one_hot)
)

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# predicted, is_correct, accuracy
predicted = tf.argmax(h, 1)
is_correct = tf.equal(predicted, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Parameters
epoch_size = 10
batch_size = 100
iterations = int(mnist.train.num_examples / batch_size)

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch_size):
        for _ in range(iterations):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            cost_v, _ = sess.run(
                [cost, optimizer], feed_dict={x: x_batch, y_one_hot: y_batch}
            )
        print("Epoch: {:2}, Cost: {:.8f}".format(epoch + 1, cost_v))

        # Evaluation
        print(
            "Accuracy: ",
            sess.run(
                accuracy,
                feed_dict={x: mnist.test.images, y_one_hot: mnist.test.labels},
            ),
        )
