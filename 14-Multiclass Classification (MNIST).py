# Logistic Regression: Multiclass Classification (MNIST)

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
nb_classes = 10

w = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

x = tf.placeholder(tf.float32, shape=[None, 784])
y_one_hot = tf.placeholder(tf.float32, shape=[None, nb_classes])

# h = exp(yi) / sigma exp(yi) = softmax
logits = tf.matmul(x, w) + b
h = tf.nn.softmax(logits)

# cost = -sigma y*log(h)
cost = tf.reduce_mean(-tf.reduce_sum(y_one_hot * tf.log(h), axis=1))

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

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
            accuracy, feed_dict={x: mnist.test.images, y_one_hot: mnist.test.labels}
        ),
    )

    # Test
    img = 22
    print("Label:", sess.run(tf.argmax(mnist.test.labels[img : img + 1], 1)))
    print(
        "Prediction:",
        sess.run(tf.argmax(h, 1), feed_dict={x: mnist.test.images[img : img + 1]}),
    )
    plt.imshow(mnist.test.images[img : img + 1].reshape(28, 28))
    plt.show()
