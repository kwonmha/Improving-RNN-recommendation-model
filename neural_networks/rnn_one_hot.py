# -*- coding: utf-8 -*-
from __future__ import print_function

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
	def __init__(self, updater=None, recurrent_layer=None, mem_frac=None, save_log=False, diversity_bias=0.0, regularization=0.0, **kwargs):
		super(RNNOneHotTF, self).__init__(**kwargs)
		
		# self.diversity_bias = np.cast[theano.config.floatX](diversity_bias)
		
		self.regularization = regularization
		self.save_log = save_log
		self.mem_frac = mem_frac
		self.updater = updater
		self.recurrent_layer = recurrent_layer

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.mem_frac)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.framework = 'tf'

		self.name = "RNN with categorical cross entropy"


	def _get_model_filename(self, epochs):
		"""Return the name of the file to save the current model
		"""
		#filename = "rnn_cce_db"+str(self.diversity_bias)+"_r"+str(self.regularization)+"_"+self._common_filename(epochs)
		filename = "rnn_cce_" + self._common_filename(epochs) + self.framework
		return filename

	def _prepare_networks(self, n_items):

		activation = self.get_activation(self.active_f)

		self.n_items = n_items
		self.X = tf.placeholder(tf.float32, [None, self.max_length, self.n_items])
		self.Y = tf.placeholder(tf.float32, [None, self.n_items])

		self.length = self.get_length(self.X)
		self.rnn_out, _state = self.recurrent_layer(self.X, self.length, activate_f=activation)

		# batch_range = tf.range(tf.shape(self.rnn_out)[0]) # [0, 1, ... b(15)]
		# self.indices = tf.stack([batch_range, self.l2], axis=1) #[[0, l2[0]], [1, [l2[1] ...]
		# self.last_rnn = tf.gather_nd(self.rnn_out, self.indices)

		self.last_rnn = self.last_relevant(self.rnn_out, self.length)

		# The sliced output is then passed through linear layer to obtain the right output size
		if self.recurrent_layer.embedding_size and self.tying:
			#target distribution은 쓰고, weight tying은 안하는 경우를 위해 여기다 놓는다.
			emb_l = lasagne.layers.get_all_layers(l_recurrent)[2]
			self.emb_params = lasagne.layers.get_all_param_values(emb_l)[0]
			# self.rec_val = lasagne.layers.get_output(l_recurrent)
			#l_recurrent_kl = lasagne.layers.NonlinearityLayer(l_recurrent, nonlinearity=self.div_by_temperature)
			# self.rec_kl_val = lasagne.layers.get_output(l_recurrent_kl) 확인용


			if not self.recurrent_layer.only_td: #weight tying
				if not self.tying_old:
					intermediate = lasagne.layers.DenseLayer(l_recurrent, num_units=self.n_items, W=emb_l.W.T, b=None, nonlinearity=None) #Wh
					inter_b = lasagne.layers.BiasLayer(intermediate) #Wh + b
					self.l_out = lasagne.layers.NonlinearityLayer(inter_b, nonlinearity=lasagne.nonlinearities.softmax) #softmax(Wh + b)
					self.l_out_kl = lasagne.layers.NonlinearityLayer(intermediate, nonlinearity=self.softmax_temperature) #softmax (Wh / t) without bias
				else:
					#아마 예전 버전...
					self.l_out = lasagne.layers.DenseLayer(l_recurrent, num_units=self.n_items, W=emb_l.W.T, nonlinearity=lasagne.nonlinearities.softmax)
					# ref github implementation, maybe mistake, bad result
					# self.l_out = lasagne.layers.DenseLayer(l_recurrent, num_units=self.n_items, W=emb_l.W.T, b=None, nonlinearity=self.softmax_temperature)
			else:
				if not self.tying_old:
					intermediate = lasagne.layers.DenseLayer(l_recurrent, num_units=self.n_items, b=None, nonlinearity=None)  # Wh
					inter_b = lasagne.layers.BiasLayer(intermediate)  # Wh + b
					self.l_out = lasagne.layers.NonlinearityLayer(inter_b, nonlinearity=lasagne.nonlinearities.softmax)  # softmax(Wh + b)
					self.l_out_kl = lasagne.layers.NonlinearityLayer(intermediate, nonlinearity=self.softmax_temperature)  # softmax (Wh / t) without bias
				else:
					self.l_out = lasagne.layers.DenseLayer(l_recurrent, num_units=self.n_items, nonlinearity=lasagne.nonlinearities.softmax)
		else:
			self.output = tf.layers.dense(self.last_rnn, self.n_items, activation=None)

		self.softmax = tf.nn.softmax(self.output)

		# loss function
		if self.recurrent_layer.embedding_size and self.tying:
			#TODO

			#cost1 = (T.nnet.categorical_crossentropy(network_output, target)).mean() 9/30까지 잘못 씀... -> 1 hot이면 그게 그거!
			cost1 = (self.kullback_leibler_divergence(target, network_output)).mean() #그래도 결과가 다른 듯?? 테스트 필요....
			cost2 = 0
			if not self.recurrent_layer.not_target_distribution:
				print("get target distribution")
				# emb_l = lasagne.layers.get_all_layers(l_last_slice)[2]
				# self.emb_params = lasagne.layers.get_all_param_values(emb_l)[0]
				t_vectors = T.dot(target, self.emb_params ) # (16, 3416) * (3416 * 200) -> (16 * 200)
				target = T.dot(t_vectors, np.transpose(self.emb_params)) # (16, 200) * (200 * 3416) -> (16 * 3416)
				target = T.nnet.softmax(target / self.temperature) # / temperature (아마 이 과정을 거치면서 T.clip을 안해도 될 듯)
				#target distribution과 one-hot 값을 비교해보자..

				if not self.tying_old:
					output_kl = lasagne.layers.get_output(self.l_out_kl)
					# output_kl = T.clip(output_kl, 10e-8, 1)
					# target = T.clip(target, 10e-8, 1)
					# self.log = T.log(self.target / network_output) #1보다 크다 == target > network_output
					cost2 = (self.kullback_leibler_divergence(target,output_kl)).mean()
				else:
					# target = T.clip(target, 10e-8, 1)
					cost2 = (self.kullback_leibler_divergence(target, network_output)).mean()
			self.cost = cost1 + cost2 * (self.gamma * self.temperature) # gamma * temperature
			# self.cost = cost1 + cost2 * 5  # temperature만 비교하기 위해 5로 고정
		else:
			self.xent = -tf.reduce_sum(self.Y * tf.log(self.softmax))
			self.cost = tf.reduce_mean(self.xent)
			# tf.summary.histogram("cost", self.cost)

		optimizer = self.updater()
		self.training = optimizer.minimize(self.cost)

		self.init = tf.global_variables_initializer()

		if self.save_log:
			self.summary = tf.summary.merge_all()
			self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
			self.writer.add_graph(self.sess.graph)
			self.run = [self.summary, self.cost, self.training]
		else:
			self.run = [self.cost, self.training]

	def get_activation(self, name):
		if name=='tanh':
			return tf.tanh
		elif name=='relu':
			return tf.nn.relu
		elif name=='elu':
			return tf.nn.elu
		elif name=='sigmoid':
			return tf.nn.sigmoid

	def last_relevant(self, input, length):
		batch_size = tf.shape(input)[0]
		max_length = int(input.get_shape()[1])
		input_size = int(input.get_shape()[2])
		index = tf.range(0, batch_size) * max_length + (length - 1)
		flat = tf.reshape(input, [-1, input_size])
		return tf.gather(flat, index)

	def get_length(self, sequence):
		used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
		length = tf.reduce_sum(used, 1)
		length = tf.cast(length, tf.int32)
		return length

	def kullback_leibler_divergence(self, y_true, y_pred):
		# y_true = tf.clip(y_true, 10e-8, 1)
		# y_pred = tf.clip(y_pred, 10e-8, 1)
		return tf.reduce_mean(tf.reduce_sum(y_true * tf.log(y_true / y_pred), axis=-1))

	def softmax_temperature(self, x):
		return tf.nn.softmax(x / self.temperature)

	def _prepare_input(self, sequences):
		""" Sequences is a list of [user_id, input_sequence, targets]
		"""
		# print("_prepare_input()")
		batch_size = len(sequences)

		# Shape of return variables
		X = np.zeros((batch_size, self.max_length, self.n_items), dtype=self._input_type) # input of the RNN
		Y = np.zeros((batch_size, self.n_items), dtype='float32') # output target

		for i, sequence in enumerate(sequences):
			user_id, in_seq, target = sequence

			seq_features = np.array(list(map(lambda x: self._get_features(x), in_seq)))
			X[i, :len(in_seq), :] = seq_features # Copy sequences into X

			Y[i][target[0][0]] = 1.

		return X, Y

	def _compute_validation_metrics(self, metrics):
		"""
		add value to lists in metrics dictionary
		"""
		ev = evaluation.Evaluator(self.dataset, k=10)
		if not self.iter:
				for batch_input, goal in self._gen_mini_batch(self.dataset.validation_set(epochs=1), test=True):
					output = self.sess.run(self.softmax, feed_dict={self.X: batch_input[0]})
					predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
					# print("predictions")
					# print(predictions)
					ev.add_instance(goal, predictions)
		else:
			for sequence, user in self.dataset.validation_set(epochs=1):
				seq_lengths = list(range(1, len(sequence))) # 1, 2, 3, ... len(sequence)-1
				for length in seq_lengths:
					X = np.zeros((1, self.max_length, self._input_size()), dtype=self._input_type)  # input shape of the RNN
					# Y = np.zeros((1, self.n_items))  # Y가 왜 있는지????? 안쓰임

					seq_by_max_length = sequence[max(length - self.max_length, 0):length]  # last max length or all
					X[0, :len(seq_by_max_length), :] = np.array(map(lambda x: self._get_features(x), seq_by_max_length))

					output = self.sess.run(self.softmax, feed_dict={self.X: X})
					predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
					# print("predictions")
					# print(predictions)
					goal = sequence[length:][0]
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
		tf.train.Saver().restore(self.sess, filename)

