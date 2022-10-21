# Tensor Manipulation

import tensorflow as tf
import numpy as np

# TensorFlow Interactive Session
sess = tf.InteractiveSession()

# numpy
t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
print(t)
print(t.ndim)
print(t.shape)

# shape, rank, axis
t = tf.constant([1, 2, 3, 4])
print(tf.shape(t).eval())

# random
print(tf.random_normal([4]).eval())

# squeeze
print(tf.squeeze([[0], [1], [2]]).eval())

# expand dims
print(tf.expand_dims([0, 1, 2], 1).eval())

# like
t = tf.constant([[0], [1], [2]])
print(tf.ones_like(t).eval())
print(tf.zeros_like(t).eval())

# stack
x = [1, 4]
y = [2, 5]
z = [3, 6]
print(tf.stack([x, y, z]).eval())
print(tf.stack([x, y, z], axis=1).eval())

# Close TensorFlow Session
sess.close()
