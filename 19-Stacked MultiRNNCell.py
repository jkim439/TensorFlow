# Stacked MultiRNNCell

import tensorflow as tf
import numpy as np

data = "if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."
data_dic = {c: i for i, c in enumerate(list(set(data)))}
data_idx = [data_dic[c] for c in data]

# [1, 179, 25] (hidden_size: num of classes, sequence_length: except last char)
batch_size = 1
sequence_length = len(data) - 1
hidden_size = len(data_dic)

# Initialize data
x_data = [data_idx[:-1]]
y_data = [data_idx[1:]]

# Make tensor
x = tf.placeholder(tf.int32, [None, sequence_length])
y = tf.placeholder(tf.int32, [None, sequence_length])
x_one_hot = tf.one_hot(x, hidden_size)

# Create Multi RNN
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
cell_multi = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
output, _ = tf.nn.dynamic_rnn(cell_multi, x_one_hot, dtype=tf.float32)
w = tf.ones([batch_size, sequence_length])

# Fully Connected Layer
output = tf.reshape(output, [-1, hidden_size])  # [n, 25]
output = tf.contrib.layers.fully_connected(output, hidden_size, activation_fn=None)
output = tf.reshape(output, [batch_size, sequence_length, hidden_size])  # [1, 179, 25]

# Loss, Cost
loss = tf.reduce_mean(
    tf.contrib.seq2seq.sequence_loss(logits=output, targets=y, weights=w)
)
cost = tf.reduce_mean(tf.square(loss))

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

# Parameters
epoch_size = 101

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch_size):
        cost_v, outputv, _ = sess.run(
            [cost, output, optimizer], feed_dict={x: x_data, y: y_data}
        )
        print("Training...", str(epoch) + "/" + str(epoch_size - 1), "\tCost: ", cost_v)

    for i in outputv:
        print("".join([list(data_dic)[j] for j in np.argmax(i, 1)]))
