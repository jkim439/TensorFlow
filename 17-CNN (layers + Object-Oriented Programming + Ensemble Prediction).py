# CNN (layers + Object-Oriented Programming + Ensemble Prediction)

import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.build()

    def build(self):
        with tf.variable_scope(self.name):
            self.x = tf.placeholder(tf.float32, shape=[None, 784])  # n x 784
            self.y_one_hot = tf.placeholder(tf.float32, shape=[None, 10])
            img = tf.reshape(self.x, [-1, 28, 28, 1])  # 28 x 28 x 1color x nimg

            self.training = tf.placeholder(tf.bool)

            # CNN Layer 1: CONV - ReLU - POOL - DROP
            conv1 = tf.layers.conv2d(
                inputs=img,
                filters=32,
                kernel_size=[3, 3],
                padding="SAME",
                activation=tf.nn.relu,
            )
            pool1 = tf.layers.max_pooling2d(
                inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2
            )
            drop1 = tf.layers.dropout(inputs=pool1, rate=0.5, training=self.training)

            # CNN Layer 2: CONV - ReLU - POOL - DROP
            conv2 = tf.layers.conv2d(
                inputs=drop1,
                filters=64,
                kernel_size=[3, 3],
                padding="SAME",
                activation=tf.nn.relu,
            )
            pool2 = tf.layers.max_pooling2d(
                inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2
            )
            drop2 = tf.layers.dropout(inputs=pool2, rate=0.5, training=self.training)

            # Fully Connected (FC), Dense Layer
            flat = tf.reshape(drop2, [-1, 7 * 7 * 64])
            w = tf.get_variable(
                "w",
                shape=[7 * 7 * 64, 10],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            b = tf.Variable(tf.random_normal([10]))

            # h = exp(yi) / sigma exp(yi) = softmax
            logits = tf.matmul(flat, w) + b
            self.h = tf.nn.softmax(logits)

            # cost = sigma y*log(h) = cross-entropy
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=self.y_one_hot
                )
            )

            # Adam Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
                self.cost
            )

            # predicted, is_correct, accuracy
            predicted = tf.argmax(self.h, 1)
            is_correct = tf.equal(predicted, tf.argmax(self.y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def train(self, x_data, y_data, training=True):
        return self.sess.run(
            [self.cost, self.optimizer],
            feed_dict={self.x: x_data, self.y_one_hot: y_data, self.training: training},
        )

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(
            self.accuracy,
            feed_dict={self.x: x_test, self.y_one_hot: y_test, self.training: training},
        )

    def predict(self, x_test, training=False):
        return self.sess.run(
            self.h, feed_dict={self.x: x_test, self.training: training}
        )


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# TensorFlow Session
with tf.Session() as sess:

    # Models
    models = []
    num_models = 2
    for m in range(num_models):
        models.append(Model(sess, "model" + str(m)))

    # Parameters
    epoch_size = 10
    batch_size = 100
    iterations = int(mnist.train.num_examples / batch_size)

    # It should run after model is created
    sess.run(tf.global_variables_initializer())

    # Training
    costs = [0] * num_models
    for epoch in range(epoch_size):
        for _ in range(iterations):
            x_batch, y_batch = mnist.train.next_batch(batch_size)

            for i, m in enumerate(models):
                cost_v, _ = m.train(x_batch, y_batch)
                costs[i] = cost_v

        print(
            "Epoch: {:2}, Cost: {}".format(epoch + 1, ["%.4f" % item for item in costs])
        )

    # Get accuracy
    predictions = np.zeros([len(mnist.test.labels), 10])
    for m in models:
        print(m.name, "Accuracy:", m.get_accuracy(mnist.test.images, mnist.test.labels))
        predictions += m.predict(mnist.test.images)

    # Ensemble
    ensemble_is_correct = tf.equal(
        tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1)
    )
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_is_correct, tf.float32))
    print("Ensemble Accuracy: ", sess.run(ensemble_accuracy))

# Epoch:  1, Cost: ['0.1332', '0.1435']
# Epoch:  2, Cost: ['0.0807', '0.0662']
# Epoch:  3, Cost: ['0.0685', '0.0778']
# Epoch:  4, Cost: ['0.0309', '0.0561']
# Epoch:  5, Cost: ['0.0313', '0.0339']
# Epoch:  6, Cost: ['0.0309', '0.0643']
# Epoch:  7, Cost: ['0.0730', '0.0239']
# Epoch:  8, Cost: ['0.0311', '0.0580']
# Epoch:  9, Cost: ['0.0674', '0.1068']
# Epoch: 10, Cost: ['0.0611', '0.0415']
# model0 Accuracy: 0.9896
# model1 Accuracy: 0.9883
# Ensemble Accuracy:  0.9903
