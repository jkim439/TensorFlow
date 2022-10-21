# Multiple Linear Regression (Queue Runner)

import tensorflow as tf

# Queue
queue = tf.train.string_input_producer(["data/1.csv"], shuffle=False)

# Reader
reader = tf.TextLineReader()
key, value = reader.read(queue)

# Decoder
xy = tf.decode_csv(value, record_defaults=[[0.0], [0.0], [0.0], [0.0]])

# Batch
x_data, y_data = tf.train.batch([xy[:-1], xy[-1:]], batch_size=10)

# h = x1w1 + x2w2+ x3w3 + b
w = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
h = tf.matmul(x, w) + b

# cost = 1/m sigma (h-y)^2
cost = tf.reduce_mean(tf.square(h - y))

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7).minimize(cost)

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Coord/Threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(9001):
        x_batch, y_batch = sess.run([x_data, y_data])
        sess.run([optimizer], feed_dict={x: x_batch, y: y_batch})

    print("My score will be ", sess.run(h, feed_dict={x: [[70, 73, 78]]}))  # 148

    # Coord/Threads
    coord.request_stop()
    coord.join(threads)
