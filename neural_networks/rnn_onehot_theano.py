# -*- coding: utf-8 -*-
from __future__ import print_function

import _pickle as cPickle
from helpers import evaluation
from .rnn_base import RNNBase
import lasagne
import numpy as np
import theano
import theano.tensor as T


class RNNOneHotTH(RNNBase):
	"""RNNOneHot are recurrent neural networks that do not depend on the factorization: they are based on one-hot encoding.

The parameters specific to the RNNOneHot are:
diversity_bias: a float in [0, inf) that tunes how the cost function of the network is biased towards less seen movies.
In practice, the classification error given by the categorical cross-entropy is divided by exp(diversity_bias * popularity (on a scale from 1 to 10)).
This will reduce the error associated to movies with a lot of views, putting therefore more importance on the ability of the network to correctly predict the rare movies.
A diversity_bias of 0 produces the normal behavior, with no bias.
	"""

	def __init__(self, updater=None, recurrent_layer=None, diversity_bias=0.0, regularization=0.0, **kwargs):
		super(RNNOneHotTH, self).__init__(**kwargs)

		self.diversity_bias = np.cast[theano.config.floatX](diversity_bias)

		self.regularization = regularization
		self.updater = updater
		self.recurrent_layer = recurrent_layer
		self.framework = 'th'

		self.name = "RNN with categorical cross entropy"

	def _get_model_filename(self, epochs):
		"""Return the name of the file to save the current model
		"""
		filename = "rnn_cce_db" + "_" + self._common_filename(epochs) + "." + self.framework
		return filename

	def prepare_networks(self, n_items):
		"""Prepares the building blocks of the RNN, but does not compile them:
		self.l_in : input layer
		self.l_mask : mask of the input layer
		self.target : target of the network
		self.l_out : output of the network
		self.cost : cost function
		"""

		self.n_items = n_items
		# The input is composed of two parts : the one-hot encoding of the movie, and the features of the movie
		self.l_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_length, self._input_size()))
		# The input is completed by a mask to inform the LSTM of the length of the sequence
		self.l_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_length))

		# recurrent layer
		self.l_recurrent = self.recurrent_layer(self.l_in, self.l_mask)

		# Theano tensor for the targets
		target = T.fmatrix('target_output')
		# target_popularity = T.fvector('target_popularity')
		# self.exclude = T.fmatrix('excluded_items')
		# ratings = T.fvector('ratings')

		# self.theano_inputs = [self.l_in.input_var, self.l_mask.input_var, target, target_popularity, self.exclude]  # ratings
		self.theano_inputs = [self.l_in.input_var, self.l_mask.input_var, target]  # ratings

		# The sliced output is then passed through linear layer to obtain the right output size
		if self.recurrent_layer.embedding_size and self.tying:
			# target distribution은 쓰고, weight tying은 안하는 경우를 위해 여기다 놓는다.
			emb_l = lasagne.layers.get_all_layers(l_recurrent)[2]
			self.emb_params = lasagne.layers.get_all_param_values(emb_l)[0]
			# self.rec_val = lasagne.layers.get_output(l_recurrent)
			# l_recurrent_kl = lasagne.layers.NonlinearityLayer(l_recurrent, nonlinearity=self.div_by_temperature)
			# self.rec_kl_val = lasagne.layers.get_output(l_recurrent_kl) 확인용

			if self.attention:
				l_recurrent = AttentionLayer(l_recurrent)

			if not self.recurrent_layer.only_td:  # weight tying
				if not self.tying_old:
					intermediate = lasagne.layers.DenseLayer(l_recurrent, num_units=self.n_items, W=emb_l.W.T, b=None, nonlinearity=None)  # Wh
					inter_b = lasagne.layers.BiasLayer(intermediate)  # Wh + b
					self.l_out = lasagne.layers.NonlinearityLayer(inter_b, nonlinearity=lasagne.nonlinearities.softmax)  # softmax(Wh + b)
					self.l_out_kl = lasagne.layers.NonlinearityLayer(intermediate, nonlinearity=self.softmax_temperature)  # softmax (Wh / t) without bias
				else:
					# 아마 예전 버전...
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
		elif self.attention:
			attention = AttentionLayer(l_recurrent)
			self.l_out = lasagne.layers.DenseLayer(attention, num_units=self.n_items, nonlinearity=lasagne.nonlinearities.softmax)
		else:
			self.l_out = lasagne.layers.DenseLayer(self.l_recurrent, num_units=self.n_items, nonlinearity=lasagne.nonlinearities.softmax)

		# lasagne.layers.get_output produces a variable for the output of the net
		network_output = lasagne.layers.get_output(self.l_out)
		# print(lasagne.layers.get_output_shape(self.l_out)) #(16, 3706)

		# loss function
		# self.cost = (T.nnet.categorical_crossentropy(network_output,self.target) / self.target_popularity).mean()
		if self.recurrent_layer.embedding_size and self.tying:
			# cost1 = (T.nnet.categorical_crossentropy(network_output, target)).mean() 9/30까지 잘못 씀... -> 1 hot이면 그게 그거!
			cost1 = (self.kullback_leibler_divergence(target, network_output)).mean()  # 그래도 결과가 다른 듯?? 테스트 필요....
			cost2 = 0
			if not self.recurrent_layer.not_target_distribution:
				print("get target distribution")
				# emb_l = lasagne.layers.get_all_layers(l_last_slice)[2]
				# self.emb_params = lasagne.layers.get_all_param_values(emb_l)[0]
				t_vectors = T.dot(target, self.emb_params)  # (16, 3416) * (3416 * 200) -> (16 * 200)
				target = T.dot(t_vectors, np.transpose(self.emb_params))  # (16, 200) * (200 * 3416) -> (16 * 3416)
				target = T.nnet.softmax(target / self.temperature)  # / temperature (아마 이 과정을 거치면서 T.clip을 안해도 될 듯)
				# target distribution과 one-hot 값을 비교해보자..

				if not self.tying_old:
					output_kl = lasagne.layers.get_output(self.l_out_kl)
					# output_kl = T.clip(output_kl, 10e-8, 1)
					# target = T.clip(target, 10e-8, 1)
					# self.log = T.log(self.target / network_output) #1보다 크다 == target > network_output
					cost2 = (self.kullback_leibler_divergence(target, output_kl)).mean()
				else:
					# target = T.clip(target, 10e-8, 1)
					cost2 = (self.kullback_leibler_divergence(target, network_output)).mean()
			self.cost = cost1 + cost2 * (self.gamma * self.temperature)  # gamma * temperature
		# self.cost = cost1 + cost2 * 5  # temperature만 비교하기 위해 5로 고정
		else:
			# self.cost = (ratings * T.nnet.categorical_crossentropy(network_output, target)).mean()
			self.cost = (-T.sum(target * T.log(network_output), axis=network_output.ndim - 1)).mean()

		if self.regularization > 0.:
			self.cost += self.regularization * lasagne.regularization.l2(self.l_out.b)
		# self.cost += self.regularization * lasagne.regularization.regularize_layer_params(self.l_out, lasagne.regularization.l2)
		elif self.regularization < 0.:
			self.cost -= self.regularization * lasagne.regularization.l1(self.l_out.b)
		# self.cost -= self.regularization * lasagne.regularization.regularize_layer_params(self.l_out, lasagne.regularization.l1)

	def kullback_leibler_divergence(self, y_true, y_pred):
		y_true = T.clip(y_true, 10e-8, 1)
		y_pred = T.clip(y_pred, 10e-8, 1)
		return T.sum(y_true * T.log(y_true / y_pred), axis=-1)

	def softmax_temperature(self, x):
		return T.nnet.softmax(x / self.temperature)

	def _prepare_input(self, sequences):
		""" Sequences is a list of [user_id, input_sequence, targets]
		"""
		# print("_prepare_input()")
		batch_size = len(sequences)

		# Shape of return variables
		X = np.zeros((batch_size, self.max_length, self._input_size()), dtype=self._input_type)  # input of the RNN
		mask = np.zeros((batch_size, self.max_length))  # mask of the input (to deal with sequences of different length)
		Y = np.zeros((batch_size, self.n_items))  # output target
		# pop = np.zeros((batch_size,))  # output target
		# exclude = np.zeros((batch_size, self.n_items), dtype=theano.config.floatX)
		# ratings = np.ones((batch_size,))

		for i, sequence in enumerate(sequences):
			user_id, in_seq, target = sequence

			seq_features = np.array(list(map(lambda x: self._get_features(x), in_seq)))
			X[i, :len(in_seq), :] = seq_features  # Copy sequences into X
			mask[i, :len(in_seq)] = 1

			Y[i][target[0][0]] = 1

			# pop[i] = self.dataset.item_popularity[target[0][0]] ** self.diversity_bias
			# exclude[i, [j[0] for j in in_seq]] = 1

		# return X, mask.astype(theano.config.floatX), Y, pop.astype(theano.config.floatX), exclude
		return X, mask.astype(theano.config.floatX), Y

	def _save(self, filename):
		super(RNNOneHotTH, self)._save(filename)
		param = lasagne.layers.get_all_param_values(self.l_out)
		f = open(filename, 'wb')
		cPickle.dump(param, f, protocol=0)
		f.close()

	def _load(self, filename):
		'''Load parameters values from a file
		'''
		f = open(filename, 'rb')
		param = cPickle.load(f)
		f.close()
		#print(param[0].astype(theano.config.floatX).shape) 기존 파일들을 불러와서 shape가 안 맞는듯
		lasagne.layers.set_all_param_values(self.l_out, [i.astype(theano.config.floatX) for i in param])

	def _compile_train_function(self):
		''' Compile self.train.
		self.train recieves a sequence and a target for every steps of the sequence,
		compute error on every steps, update parameter and return global cost (i.e. the error).
		'''
		print("Compiling train...")
		all_params = lasagne.layers.get_all_params(self.l_out, trainable=True)
		updates = self.updater(self.cost, all_params)

		deterministic_output = lasagne.layers.get_output(self.l_out, deterministic=True)
		rnn_output = lasagne.layers.get_output(self.l_recurrent, deterministic=True)
		# Compile network
		self.train_function = theano.function(self.theano_inputs, [self.cost, deterministic_output, rnn_output], updates=updates, allow_input_downcast=True,
											  name="Train_function", on_unused_input='ignore')
		print("Compilation done.")

	def _compile_predict_function(self):
		''' Compile self.predict, the deterministic rnn that output the prediction at the end of the sequence
		'''
		print("Compiling predict...")
		deterministic_output = lasagne.layers.get_output(self.l_out, deterministic=True)
		self.predict_function = theano.function([self.l_in.input_var, self.l_mask.input_var], deterministic_output,
												allow_input_downcast=True, name="Predict_function")
		print("Compilation done.")

	def _compile_test_function(self):
		''' Compile self.test_function, the deterministic rnn that output the k best scoring items
		'''
		print("Compiling test...")
		deterministic_output = lasagne.layers.get_output(self.l_out, deterministic=True)

		theano_test_function = theano.function([self.l_in.input_var, self.l_mask.input_var], deterministic_output, allow_input_downcast=True,
											   name="Test_function", on_unused_input='ignore')

		def precision_test_function(theano_inputs, k=10):
			output = theano_test_function(*theano_inputs)
			# print("output")
			# print(output[0])

			ids = np.argpartition(-output, range(k), axis=-1)[0, :k] #[1106 2374  253 2651  593  579 1848  802  575 1104]
			# print(ids)
			return ids

		self.test_function = precision_test_function

		print("Compilation done.")

	def _compute_validation_metrics(self, metrics):
		"""
		add value to lists in metrics dictionary
		"""
		ev = evaluation.Evaluator(self.dataset, k=10)
		if not self.iter:
				for batch_input, goal in self._gen_mini_batch(self.dataset.validation_set(epochs=1), test=True):
					predictions = self.test_function([batch_input[0], batch_input[1]])

					# print("predictions")
					# print(predictions)
					ev.add_instance(goal, predictions)
		else:
			for sequence, user in self.dataset.validation_set(epochs=1):
				seq_lengths = list(range(1, len(sequence))) # 1, 2, 3, ... len(sequence)-1
				for length in seq_lengths:
					X = np.zeros((1, self.max_length, self._input_size()), dtype=self._input_type)  # input shape of the RNN
					mask = np.zeros((1, self.max_length))  # mask of the input (to deal with sequences of different length)

					seq_by_max_length = sequence[max(length - self.max_length, 0):length]  # last max length or all
					X[0, :len(seq_by_max_length), :] = np.array(map(lambda x: self._get_features(x), seq_by_max_length))
					mask[0, :len(seq_by_max_length)] = 1

					predictions = self.test_function((X, mask.astype(theano.config.float)))
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
