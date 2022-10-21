# Logistic Regression: Multiclass Classification (Improved MNIST)
# Extra parts: nn, relu, adam, xavier, dropout

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
nb_classes = 10

x = tf.placeholder(tf.float32, shape=[None, 784])
y_one_hot = tf.placeholder(tf.float32, shape=[None, nb_classes])

keep_prob = tf.placeholder(tf.float32)

# Layer 1
with tf.name_scope("layer1"):

    # Initial Weight: Xavier Initialization (사비에르 초기화)
    w1 = tf.get_variable(
        "w1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer()
    )

    b1 = tf.Variable(tf.random_normal([512]))

    # Avoid Vanishing Gradient: ReLU instead of softmax
    l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    # Regularization: Dropout
    l1 = tf.nn.dropout(l1, keep_prob=keep_prob)

# Layer 2
with tf.name_scope("layer2"):
    w2 = tf.get_variable(
        "w2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer()
    )
    b2 = tf.Variable(tf.random_normal([512]))
    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    l2 = tf.nn.dropout(l2, keep_prob=keep_prob)

# Layer 3
with tf.name_scope("layer3"):
    w3 = tf.get_variable(
        "w3",
        shape=[512, nb_classes],
        initializer=tf.contrib.layers.xavier_initializer(),
    )
    b3 = tf.Variable(tf.random_normal([nb_classes]))

# h = exp(yi) / sigma exp(yi) = softmax
logits = tf.matmul(l2, w3) + b3
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
                [cost, optimizer],
                feed_dict={x: x_batch, y_one_hot: y_batch, keep_prob: 0.7},
            )
        print("Epoch: {:2}, Cost: {:.8f}".format(epoch + 1, cost_v))

    # Evaluation
    print(
        "Accuracy: ",
        sess.run(
            accuracy,
            feed_dict={
                x: mnist.test.images,
                y_one_hot: mnist.test.labels,
                keep_prob: 1.0,
            },
        ),
    )

    # Test
    img = 22
    print("Label:", sess.run(tf.argmax(mnist.test.labels[img : img + 1], 1)))
    print(
        "Prediction:",
        sess.run(
            tf.argmax(h, 1),
            feed_dict={x: mnist.test.images[img : img + 1], keep_prob: 1.0},
        ),
    )
    plt.imshow(mnist.test.images[img : img + 1].reshape(28, 28))
    plt.show()
