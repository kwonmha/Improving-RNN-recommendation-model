# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from .rnn_base import RNNBase
from helpers import evaluation


class RNNOneHotTF(RNNBase):
	"""RNNOneHot are recurrent neural networks that do not depend on the factorization: they are based on one-hot encoding.

The parameters specific to the RNNOneHot are:
diversity_bias: a float in [0, inf) that tunes how the cost function of the network is biased towards less seen movies.
In practice, the classification error given by the categorical cross-entropy is divided by exp(diversity_bias * popularity (on a scale from 1 to 10)).
This will reduce the error associated to movies with a lot of views, putting therefore more importance on the ability of the network to correctly predict the rare movies.
A diversity_bias of 0 produces the normal behavior, with no bias.
	"""
	def __init__(self, updater=None, recurrent_layer=None, mem_frac=None, save_log=False, log_dir=None, diversity_bias=0.0, regularization=0.0, **kwargs):
		super(RNNOneHotTF, self).__init__(**kwargs)
		
		# self.diversity_bias = np.cast[theano.config.floatX](diversity_bias)
		
		self.updater = updater
		self.recurrent_layer = recurrent_layer
		self.mem_frac = mem_frac
		self.save_log = save_log
		self.log_dir = log_dir
		self.regularization = regularization

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.mem_frac)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.framework = 'tf'

		self.name = "RNN with categorical cross entropy"

	def _get_model_filename(self, epochs):
		"""Return the name of the file to save the current model
		"""
		#filename = "rnn_cce_db"+str(self.diversity_bias)+"_r"+str(self.regularization)+"_"+self._common_filename(epochs)
		filename = "rnn_cce_" + self._common_filename(epochs) + "." + self.framework
		return filename

	def prepare_networks(self, n_items):

		activation = self.get_activation(self.active_f)

		self.n_items = n_items
		if self.recurrent_layer.embedding_size > 0:
			self.X = tf.placeholder(tf.int32, [None, self.max_length])
			word_embeddings = tf.get_variable("word_embeddings", [self.n_items, self.recurrent_layer.embedding_size])
			rnn_input = tf.nn.embedding_lookup(word_embeddings, self.X)
			# print(rnn_input) # (?, max_length, embedding_size)
		else:
			self.X = tf.placeholder(tf.float32, [None, self.max_length, self.n_items])
			rnn_input = self.X
		self.Y = tf.placeholder(tf.float32, [None, self.n_items])
		self.length = tf.placeholder(tf.int32, [None, ])

		# self.length = self.get_length(rnn_input)
		self.rnn_out, _state = self.recurrent_layer(rnn_input, self.length, activate_f=activation) #(B, max_length, hidden)

		if self.attention:
			att_mat_param = tf.get_variable("att_mat_param", [self.recurrent_layer.layers[-1], self.recurrent_layer.layers[-1]])
			att_vec_param = tf.get_variable("att_vec_param", [self.recurrent_layer.layers[-1]])
			# sim_mat = tf.tensordot(self.rnn_out, att_mat_param, axes=1) #(B,T,D)*(D,D)=>(B,T,D), undefined shape : np.shape(sim_mat) = <unknown>
			sim_mat = tf.einsum('aij,jk->aik', self.rnn_out, att_mat_param) # shape defined : np.shape(sim_mat) = (?, D)
			# print(np.shape(rnn_attention))
			sim_mat_tan = tf.tanh(sim_mat)
			# att_vec = tf.tensordot(sim_mat_tan, tf.transpose(att_vec_param), axes=1) # (B,T,D) * (D, 1)=> (B, T)
			att_vec = tf.einsum('aij,j->ai', sim_mat_tan, tf.transpose(att_vec_param))
			att_sm_vec = tf.nn.softmax(att_vec)
			rnn_attention = tf.einsum('aij,ai->aj', self.rnn_out, att_sm_vec) #(B, T, D) * (B, T) => (B, D)
		else:
			self.last_rnn = self.last_relevant(self.rnn_out, self.length)

		self.last_hidden = rnn_attention if self.attention else self.last_rnn

		# self.last_hidden = self.last_rnn

		if self.recurrent_layer.embedding_size and self.tying:

			if self.recurrent_layer.no_td: #not using embedding matrix to get new target vectors
				target = self.Y
			else: #default
				t_vectors = tf.matmul(self.Y, word_embeddings)  # (16, 3416) * (3416 * 200) -> (16 * 200)
				# print(np.shape(t_vectors)) #(?, 200)
				target = tf.matmul(t_vectors, tf.transpose(word_embeddings))  # (16, 200) * (200 * 3416) -> (16 * 3416)
				target = tf.nn.softmax(target / self.temperature)

			if self.recurrent_layer.no_wt: # not using embedding matrix to get network output
				w_nwt = tf.get_variable("w_nwt", [self.recurrent_layer.embedding_size, self.n_items])
				wh = tf.matmul(self.last_hidden, w_nwt)

			else: #default
				with tf.variable_scope(tf.get_variable_scope(), reuse=True):
					output_embeddings = tf.get_variable("word_embeddings", [self.n_items, self.recurrent_layer.embedding_size] )
					wh = tf.matmul(self.last_hidden, tf.transpose(output_embeddings))  # (16 * emb) * (emb * n) -> (16 * n)
				# wh = tf.matmul(self.last_hidden, tf.transpose(word_embeddings)) # (16 * emb) * (emb * n) -> (16 * n)

			bias = tf.get_variable("bias", self.n_items)
			whb = tf.nn.bias_add(wh, bias)  # Wh + b
			self.softmax = tf.nn.softmax(whb)
			self.cost1 = self.kullback_leibler_divergence(self.Y, self.softmax)

			if self.tying_new:
				self.cost2 = self.kullback_leibler_divergence(target, self.softmax)

			else: #following tying matrix paper
				new_output = tf.nn.softmax(wh/self.temperature)
				self.cost2 = self.kullback_leibler_divergence(target, new_output)

			self.cost = self.cost1 + self.cost2 * (self.gamma * self.temperature)  # gamma * temperature

		else:
			self.output = tf.layers.dense(self.last_hidden, self.n_items, activation=None)
			# applying attention makes last_hidden have undefined rank. need to reshape

			self.softmax = tf.nn.softmax(self.output)
			self.xent = -tf.reduce_sum(self.Y * tf.log(self.softmax))
			self.cost = tf.reduce_mean(self.xent)
			# tf.summary.histogram("cost", self.cost)

		optimizer = self.updater()
		self.training = optimizer.minimize(self.cost)

		self.init = tf.global_variables_initializer()

		if self.save_log:
			self.summary = tf.summary.merge_all()
			self.writer = tf.summary.FileWriter('./' + self.log_dir, self.sess.graph)
			self.writer.add_graph(self.sess.graph)

	def get_activation(self, name):
		if name=='tanh':
			return tf.tanh
		elif name=='relu':
			return tf.nn.relu
		elif name=='elu':
			return tf.nn.elu
		elif name=='sigmoid':
			return tf.nn.sigmoid

	def last_relevant(self, seq, length): #https://danijar.com/variable-sequence-lengths-in-tensorflow/
		batch_size = tf.shape(seq)[0]
		max_length = int(seq.get_shape()[1])
		input_size = int(seq.get_shape()[2])
		index = tf.range(0, batch_size) * max_length + (length - 1)
		flat = tf.reshape(seq, [-1, input_size])
		return tf.gather(flat, index)

	# def get_length(self, sequence):
	# 	used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
	# 	length = tf.reduce_sum(used, 1)
	# 	length = tf.cast(length, tf.int32)
	# 	return length

	def kullback_leibler_divergence(self, y_true, y_pred):
		y_true = tf.maximum(y_true, 10e-8) # prevent 0 into tf.log
		return tf.reduce_mean(tf.reduce_sum(y_true * tf.log(y_true / y_pred)))

	def _prepare_input(self, sequences):
		""" Sequences is a list of [user_id, input_sequence, targets]
		"""
		# print("_prepare_input()")
		batch_size = len(sequences)

		# Shape of return variables
		if self.recurrent_layer.embedding_size > 0:
			X = np.zeros((batch_size, self.max_length), dtype=np.int32)  # tf embedding requires movie-id sequence, not one-hot
		else:
			X = np.zeros((batch_size, self.max_length, self.n_items), dtype=self._input_type)  # input of the RNN
		length = np.zeros((batch_size), dtype=np.int32)
		Y = np.zeros((batch_size, self.n_items), dtype=np.float32)  # output target

		for i, sequence in enumerate(sequences):
			user_id, in_seq, target = sequence

			if self.recurrent_layer.embedding_size > 0:
				X[i, :len(in_seq)] = np.array([item[0] for item in in_seq])
			else:
				seq_features = np.array(list(map(lambda x: self._get_features(x), in_seq)))
				X[i, :len(in_seq), :] = seq_features  # Copy sequences into X

			length[i] = len(in_seq)
			Y[i][target[0][0]] = 1.

		return X, Y, length

	def _compute_validation_metrics(self, metrics):
		"""
		add value to lists in metrics dictionary
		"""
		ev = evaluation.Evaluator(self.dataset, k=10)
		if not self.iter:
				for batch_input, goal in self._gen_mini_batch(self.dataset.validation_set(epochs=1), test=True):
					output = self.sess.run(self.softmax, feed_dict={self.X: batch_input[0], self.length: batch_input[2]})
					predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
					# print("predictions")
					# print(predictions)
					ev.add_instance(goal, predictions)
		else:
			for sequence, user in self.dataset.validation_set(epochs=1):
				seq_lengths = list(range(1, len(sequence))) # 1, 2, 3, ... len(sequence)-1
				for seq_length in seq_lengths:
					X = np.zeros((1, self.max_length, self._input_size()), dtype=self._input_type)  # input shape of the RNN
					# Y = np.zeros((1, self.n_items))  # Y가 왜 있는지????? 안쓰임
					length = np.zeros((1,), dtype=np.int32)

					seq_by_max_length = sequence[max(length - self.max_length, 0):seq_length]  # last max length or all
					X[0, :len(seq_by_max_length), :] = np.array(map(lambda x: self._get_features(x), seq_by_max_length))
					length[0] = len(seq_by_max_length)

					output = self.sess.run(self.softmax, feed_dict={self.X: X, self.length: length})
					predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
					# print("predictions")
					# print(predictions)
					goal = sequence[seq_length:][0]
					ev.add_instance(goal, predictions)

		metrics['recall'].append(ev.average_recall())
		metrics['sps'].append(ev.sps())
		metrics['precision'].append(ev.average_precision())
		metrics['ndcg'].append(ev.average_ndcg())
		metrics['user_coverage'].append(ev.user_coverage())
		metrics['item_coverage'].append(ev.item_coverage())
		metrics['blockbuster_share'].append(ev.blockbuster_share())

		# del ev
		ev.instances = []

		return metrics

	def _save(self, filename):
		super(RNNOneHotTF, self)._save(filename)
		tf.train.Saver().save(self.sess, filename)

	def _load(self, filename):
		'''Load parameters values from a file
		'''

		saver = tf.train.Saver()
		# saver.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(filename))) #i don't need checkpoint file now
		saver.restore(self.sess, filename[:-5]) #remove ".meta"
		print("load success")

