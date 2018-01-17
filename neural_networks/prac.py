# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Training Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 50 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
	'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
	'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, timesteps, n_input)
	# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	# x = tf.unstack(x, timesteps, 1)

	# Define a lstm cell with tensorflow
	lstm_cell = rnn.LSTMCell(num_hidden, forget_bias=1.0)

	# Get lstm cell output
	outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
	# print(outputs)
	# exit()

	# Linear activation, using rnn inner loop last output
	# return tf.matmul(outputs[-1], weights['out']) + biases['out']
	return tf.layers.dense(outputs[:, -1], num_classes, activation=None)

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=logits, labels=Y))
# loss_op = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction), 1))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

	# Run the initializer
	sess.run(init)

	for step in range(1, training_steps+1):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		# Reshape data to get 28 seq of 28 elements
		batch_x = batch_x.reshape((batch_size, timesteps, num_input))
		# Run optimization op (backprop)
		sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
		if step % display_step == 0 or step == 1:
			# Calculate batch loss and accuracy
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
																 Y: batch_y})
			print("Step " + str(step) + ", Minibatch Loss= " + \
				  "{:.4f}".format(loss) + ", Training Accuracy= " + \
				  "{:.3f}".format(acc))

	print("Optimization Finished!")

	# Calculate accuracy for 128 mnist test images
	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
	test_label = mnist.test.labels[:test_len]
	print("Testing Accuracy:", \
		sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))


#======================================================================================================================
# lr = 0.001
# steps = 10000
# batch_size = 128
#
# num_input = 28
# timesteps = 28
# num_hidden = [128]
# num_classes = 10
#
# seq_length = [28] * batch_size
#
# X = tf.placeholder(tf.float32, [None, timesteps, num_input])
# Y = tf.placeholder(tf.float32, [None, num_classes])
#
# x = tf.unstack(X, timesteps, 1)
# rnn_cell = tf.nn.rnn_cell.LSTMCell
# cells = [rnn_cell(h, initializer=tf.random_uniform_initializer(), activation=tf.tanh, state_is_tuple=True) for
# 		 _, h in enumerate(num_hidden)]
# multi_cells = rnn.MultiRNNCell(cells, state_is_tuple=True)
# rnn_outputs, _states = tf.nn.static_rnn(multi_cells, x, dtype=tf.float32)
#
# output = tf.layers.dense(rnn_outputs[-1], num_classes, activation=None)
# prediction = tf.nn.softmax(output)
#
# loss_op = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction)))
# optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
# train_op = optimizer.minimize(loss_op)
#
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# # Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()
#
# # Start training
# with tf.Session() as sess:
#
#     # Run the initializer
#     sess.run(init)
#
#     for step in range(1, steps+1):
#         batch_x, batch_y = mnist.train.next_batch(batch_size)
#         # Reshape data to get 28 seq of 28 elements
#         batch_x = batch_x.reshape((batch_size, timesteps, num_input))
#         # Run optimization op (backprop)
#         sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
#         if step % 200 == 0 or step == 1:
#             # Calculate batch loss and accuracy
#             loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
#                                                                  Y: batch_y})
#             print("Step " + str(step) + ", Minibatch Loss= " + \
#                   "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.3f}".format(acc))
#
#     print("Optimization Finished!")
#
#     # Calculate accuracy for 128 mnist test images
#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
#     test_label = mnist.test.labels[:test_len]
#     print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))


#======================================================================================================================
# X = tf.placeholder(tf.float32, [2, 3])
# Y = tf.placeholder(tf.float32, [2, 3])
#
# x = np.array([[-1, 0, 1], [-1, 0, 1]])
# y = np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
#
# softmax = tf.nn.softmax(X)
# xent_ = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(softmax), 1))
#
# session = tf.Session()
#
# xent = session.run(xent_, feed_dict={X:x, Y:y})
# print(xent)

#======================================================================================================================

# X = tf.placeholder(tf.float32, [2, 3, 4])
# x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
# 				[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
#
# XT = tf.transpose(X, [1, 0, 2])
# XT_last = tf.gather(XT, 1)
#
# sess = tf.Session()
#
# last, tp = sess.run([XT_last, XT], feed_dict={X:x})
# print(tp)
# print("===============================")
# print(XT.get_shape())
# print(last)
#
# print(XT)