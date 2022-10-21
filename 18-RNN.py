# Recurrent Neural Network (RNN)

import tensorflow as tf
import numpy as np

data = "if you want you"

# {'a': 0, 'f': 1, 'u': 2, 'o': 3, 'i': 4, 'n': 5, 'w': 6, 't': 7, 'y': 8, ' ': 9}
data_dic = {c: i for i, c in enumerate(list(set(data)))}

# [4, 1, 9, 8, 3, 2, 9, 6, 0, 5, 7, 9, 8, 3, 2]
data_idx = [data_dic[c] for c in data]

# [1, 9, 10] (hidden_size: num of classes, sequence_length: except last char)
batch_size = 1
sequence_length = len(data) - 1
hidden_size = len(data_dic)

# Initialize data
x_data = [
    data_idx[:-1]
]  # except last char [[4, 1, 9, 8, 3, 2, 9, 6, 0, 5, 7, 9, 8, 3]]
y_data = [
    data_idx[1:]
]  # except first char [[1, 9, 8, 3, 2, 9, 6, 0, 5, 7, 9, 8, 3, 2]]

# Make tensor
x = tf.placeholder(tf.int32, [None, sequence_length])
y = tf.placeholder(tf.int32, [None, sequence_length])
x_one_hot = tf.one_hot(x, hidden_size)

# Create RNN
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
output, _ = tf.nn.dynamic_rnn(cell, x_one_hot, dtype=tf.float32)
w = tf.ones([batch_size, sequence_length])

# Loss
loss = tf.reduce_mean(
    tf.contrib.seq2seq.sequence_loss(logits=output, targets=y, weights=w)
)

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# predicted
predicted = tf.argmax(output, 2)

# TensorFlow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        lossv, _ = sess.run([loss, optimizer], feed_dict={x: x_data, y: y_data})

        if step % 10 == 0:
            data_predicted = [
                list(data_dic)[i]
                for i in np.squeeze(sess.run(predicted, feed_dict={x: x_data}))
            ]
            print(step, "".join(data_predicted))
