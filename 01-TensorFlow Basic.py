# TensorFlow Basic

import os
import tensorflow as tf

print(tf.__version__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 1. Build graph
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.add(node1, node2)

# 2. Feed data and run graph
sess = tf.Session()

# 3. Update variables in the graph
print(sess.run([node1, node2, node3]))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.add(a, b)
print(sess.run(c, feed_dict={a: 3.5, b: 4.5}))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.add(a, b)
print(sess.run(c, feed_dict={a: [0.5, 1.5], b: [2.5, 4.5]}))

# Close TensorFlow Session
sess.close()
