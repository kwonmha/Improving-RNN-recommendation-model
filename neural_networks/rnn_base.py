# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import os
import random
import re
import sys
from time import time

import numpy as np

from .sequence_noise import SequenceNoise
from .target_selection import SelectTargets

class RNNBase(object):
	"""Base for RNN object.
	"""
	def __init__(self,
		sequence_noise=SequenceNoise(),
		target_selection=SelectTargets(),
		active_f='tanh',
		max_length=30,
		batch_size=16,
		tying=False,
		temperature=10,
		gamma=0.5,
		iter=False,
		tying_new = False,
		attention=False):

		super(RNNBase, self).__init__()

		self.max_length = max_length
		self.batch_size = batch_size
		self.sequence_noise = sequence_noise
		self.target_selection = target_selection
		self.active_f=active_f
		self.tying = tying
		self.temperature = temperature
		self.gamma = gamma
		self.iter = iter
		self.tying_new = tying_new
		self.attention = attention

		self._input_type = 'float32'

		self.name = "RNN base"
		self.metrics = {'recall': {'direction': 1},
						'precision': {'direction': 1},
						'sps': {'direction': 1},
						'user_coverage': {'direction': 1},
						'item_coverage': {'direction': 1},
						'ndcg': {'direction': 1},
						'blockbuster_share': {'direction': -1}
						}


	def _common_filename(self, epochs):
		'''Common parts of the filename accros sub classes.
		'''
		filename = "ml"+str(self.max_length)+"_bs"+str(self.batch_size)+"_ne"+str(epochs)+"_"+self.recurrent_layer.name + "_" + self.updater.name + "_" + self.target_selection.name

		if self.sequence_noise.name != "":
			filename += "_" + self.sequence_noise.name

		if self.active_f != 'tanh':
			filename += "_act" + self.active_f[0].upper()
		if self.tying:
			filename += "_ty_tp" + str(self.temperature) + "_gm" + str(self.gamma)
			if self.tying_new:
				filename += "_new"
		if self.iter:
			filename += "_it"
		if self.attention:
			filename += "_att"
		return filename

	def top_k_recommendations(self, sequence, k=10):
		''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

		# Prepare RNN input
		X = np.zeros((1, self.max_length, self._input_size()), dtype=self._input_type) # input shape of the RNN

		seq_by_max_length = sequence[-min(self.max_length, len(sequence)):] #last max length or all
		X[0, :len(seq_by_max_length), :] = np.array(list(map(lambda x: self._get_features(x), seq_by_max_length)))

		# Run RNN
		if self.framework == 'tf':
			output = self.sess.run(self.softmax, feed_dict={self.X: X})
		elif self.framework == 'th':
			if not hasattr(self, 'predict_function'):
				self._compile_predict_function()
			mask = np.zeros((1, self.max_length))
			mask[0, :len(seq_by_max_length)] = 1
			output = self.predict_function(X, mask)
		else:
			output = self.model.predict_on_batch(X)

		# find top k according to output
		return list(np.argpartition(-output[0], list(range(k)))[:k])

	def set_dataset(self, dataset):
		self.dataset = dataset
		self.target_selection.set_dataset(dataset)

	def get_pareto_front(self, metrics, metrics_names):
		costs = np.zeros((len(metrics[metrics_names[0]]), len(metrics_names)))
		for i, m in enumerate(metrics_names):
			costs[:, i] = np.array(metrics[m]) * self.metrics[m]['direction']
		is_efficient = np.ones(costs.shape[0], dtype=bool)
		for i, c in enumerate(costs):
			if is_efficient[i]:
				is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
		return np.where(is_efficient)[0].tolist()


	def train(self, dataset,
		max_time=np.inf,
		progress=2.0, 
		autosave='All',
		save_dir='', 
		min_iterations=0, 
		max_iter=np.inf,
		load_last_model=False,
		early_stopping=None,
		validation_metrics=['sps']):
		'''Train the model based on the sequence given by the training_set

		max_time is used to set the maximumn amount of time (in seconds) that the training can last before being stop.
			By default, max_time=np.inf, which means that the training will last until the training_set runs out, or the user interrupt the program.
		
		progress is used to set when progress information should be printed during training. It can be either an int or a float:
			integer : print at linear intervals specified by the value of progress (i.e. : progress, 2*progress, 3*progress, ...)
			float : print at geometric intervals specified by the value of progress (i.e. : progress, progress^2, progress^3, ...)

		max_progress_interval can be used to have geometric intervals in the begining then switch to linear intervals. 
			It ensures, independently of the progress parameter, that progress is shown at least every max_progress_interval.

		time_based_progress is used to choose between using number of iterations or time as a progress indicator. True means time (in seconds) is used, False means number of iterations.

		autosave is used to set whether the model should be saved during training. It can take several values:
			All : the model will be saved each time progress info is printed.
			Best : save only the best model so far
			None : does not save

		min_iterations is used to set a minimum of iterations before printing the first information (and saving the model).

		save_dir is the path to the directory where models are saved.

		load_last_model: if true, find the latest model in the directory where models should be saved, and load it before starting training.

		early_stopping : should be a callable that will recieve the list of validation error and the corresponding epochs and return a boolean indicating whether the learning should stop.
		'''

		self.dataset = dataset
		self.target_selection.set_dataset(dataset)

		if len(set(validation_metrics) & set(self.metrics.keys())) < len(validation_metrics):
			raise ValueError(
				'Incorrect validation metrics. Metrics must be chosen among: ' + ', '.join(self.metrics.keys()))

		if self.framework=='tf':
			self.sess.run(self.init)
		elif self.framework == 'th':
			if not hasattr(self, 'train_function'):
				self._compile_train_function()
			if not hasattr(self, 'test_function'):
				self._compile_test_function()

		# Load last model if needed
		iterations = 0
		epochs_offset = 0
		if load_last_model:
			epochs_offset = self.load_last(save_dir)

		# Make batch generator
		#batch_generator = threaded_generator(self._gen_mini_batch(self.sequence_noise(dataset.training_set())))

		batch_generator = self._gen_mini_batch(self.sequence_noise(self.dataset.training_set()))

		start_time = time()
		next_save = int(progress)
		#val_costs = []
		train_costs = []
		current_train_cost = []
		epochs = []
		metrics = {name: [] for name in self.metrics.keys()}
		filename = {}
		bs_sum = 0
		ts_sum = 0

		try: 
			while time() - start_time < max_time and iterations < max_iter:

				# Train with a new batch
				try:
					bs = time()
					batch = next(batch_generator)
					bs_sum += time() - bs
					# np.set_printoptions(threshold=np.inf)

					if self.framework == 'tf':
						ts = time()
						if self.save_log:
							s, cost, _ = self.sess.run([self.summary, self.cost, self.tarining], feed_dict={self.X: batch[0], self.Y: batch[1], self.length: batch[2]})
							self.writer.add_summary(s, iterations)
						else:
							cost, _ = self.sess.run([self.cost, self.training], feed_dict={self.X: batch[0], self.Y: batch[1], self.length: batch[2]})

						ts_sum += time() - ts

						# print("================================================")
						# print(rnn_out[0,:30,:6])
						# print(last_rnn[0, :6])
						# print(cost)

					elif self.framework.startswith('k'):
						# self.model.fit(batch[0], batch[2])
						cost = self.model.train_on_batch(batch[0], batch[1])
						# outputs = self.model.predict_on_batch(batch[0])
						# print(outputs[0, :6])
						# print(batch[1])

					else:
						ts = time()
						cost = self.train_function(*batch)
						ts_sum += time() - ts
						# print(output)
						# print(cost)

					# exit()

					if np.isnan(cost):
						raise ValueError("Cost is NaN")

				except StopIteration:
					break

				current_train_cost.append(cost)
				#current_train_cost.append(0)

				# Check if it is time to save the model
				iterations += 1

				if iterations >= next_save:
					if iterations >= min_iterations:
						# Save current epoch
						epochs.append(epochs_offset + self.dataset.training_set.epochs)

						# Average train cost
						train_costs.append(np.mean(current_train_cost))
						current_train_cost = []

						# Compute validation cost
						metrics = self._compute_validation_metrics(metrics)

						# Print info
						self._print_progress(iterations, epochs[-1], start_time, train_costs, metrics,
											 validation_metrics)
						print(bs_sum, ts_sum)
						bs_sum = 0
						ts_sum = 0
						# exit()

						# Save model
						run_nb = len(metrics[list(self.metrics.keys())[0]]) - 1
						if autosave == 'All':
							filename[run_nb] = save_dir + self.framework + "/" + self._get_model_filename(round(epochs[-1], 3))
							self._save(filename[run_nb])
						elif autosave == 'Best':
							pareto_runs = self.get_pareto_front(metrics, validation_metrics)
							if run_nb in pareto_runs:
								filename[run_nb] = save_dir + self.framework + "/" + self._get_model_filename(round(epochs[-1], 3))
								self._save(filename[run_nb])
								to_delete = [r for r in filename if r not in pareto_runs]
								for run in to_delete:
									try:
										if self.framework == 'tf' :
											os.remove(filename[run] + ".data-00000-of-00001")
											os.remove(filename[run] + ".index")
											os.remove(filename[run] + ".meta")
										else:
											os.remove(filename[run])
									except OSError:
										print('Warning : Previous model could not be deleted')
									del filename[run]
						# exit()

						if early_stopping is not None:
							if all([early_stopping(epochs, metrics[m]) for m in validation_metrics]):
								break

					next_save += progress

		except KeyboardInterrupt:
			print('Training interrupted')

		best_run = np.argmax(
			np.array(metrics[validation_metrics[0]]) * self.metrics[validation_metrics[0]]['direction'])
		return ({m: metrics[m][best_run] for m in self.metrics.keys()}, time() - start_time, filename[best_run])

	def _compute_validation_metrics(self, metrics):
		"""
		add value to lists in metrics dictionary
		"""


	def _gen_mini_batch(self, sequence_generator, test=False):
		''' Takes a sequence generator and produce a mini batch generator.
		The mini batch have a size defined by self.batch_size, and have format of the input layer of the rnn.

		test determines how the sequence is splitted between training and testing
			test == False, the sequence is split randomly
			test == True, the sequence is split in the middle

		if test == False, max_reuse_sequence determines how many time a single sequence is used in the same batch.
			with max_reuse_sequence = inf, one sequence will be used to make the whole batch (if the sequence is long enough)
			with max_reuse_sequence = 1, each sequence is used only once in the batch
		N.B. if test == True, max_reuse_sequence = 1 is used anyway
		'''

		while True:
			j = 0
			sequences = []
			batch_size = self.batch_size
			if test:
				batch_size = 1
			while j < batch_size: # j : user order

				sequence, user_id = next(sequence_generator)

				# finds the lengths of the different subsequences
				if not test: # training set
					seq_lengths = sorted(
						random.sample(range(2, len(sequence)), #range
									  min([self.batch_size - j, len(sequence) - 2])) #population
					)
				elif self.iter:
					batch_size = len(sequence) -1
					seq_lengths = list(range(1, len(sequence)))
				else: #validating set
					seq_lengths = [int(len(sequence) / 2)] # half of len

				skipped_seq = 0
				for l in seq_lengths:
					# target is only for rnn with hinge, logit and logsig.
					target = self.target_selection(sequence[l:], test=test)
					if len(target) == 0:
						skipped_seq += 1
						continue
					start = max(0, l - self.max_length) # sequences cannot be longer than self.max_length
					#print(target)
					sequences.append([user_id, sequence[start:l], target])
					# print([user_id, sequence[start:l], target])

				j += len(seq_lengths) - skipped_seq #?????????

			if test:
				yield self._prepare_input(sequences), [i[0] for i in sequence[seq_lengths[0]:]]
			else:
				yield self._prepare_input(sequences)


	def _print_progress(self, iterations, epochs, start_time, train_costs, metrics, validation_metrics):
		'''Print learning progress in terminal
		'''
		print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
		print("Last train cost : ", train_costs[-1])
		for m in self.metrics:
			print(m, ': ', metrics[m][-1])
			if m in validation_metrics:
				print('Best ', m, ': ',
					  max(np.array(metrics[m]) * self.metrics[m]['direction']) * self.metrics[m]['direction'])

		print('-----------------')

		# Print on stderr for easier recording of progress
		print(iterations, epochs, time() - start_time, train_costs[-1],
			  ' '.join(map(str, [metrics[m][-1] for m in self.metrics])), file=sys.stderr)

	def _get_model_filename(self, iterations):
		'''Return the name of the file to save the current model
		'''
		raise NotImplemented

	def prepare_networks(self):
		''' Prepares the building blocks of the RNN, but does not compile them:
		self.l_in : input layer
		self.l_mask : mask of the input layer
		self.target : target of the network
		self.l_out : last output of the network
		self.cost : cost function

		and maybe others
		'''
		raise NotImplemented
		
	def _compile_train_network(self):
		''' Compile self.train. 
		self.train recieves a sequence and a target for every steps of the sequence, 
		compute error on every steps, update parameter and return global cost (i.e. the error).
		'''
		raise NotImplemented

	def _compile_predict_network(self):
		''' Compile self.predict, the deterministic rnn that output the prediction at the end of the sequence
		'''
		raise NotImplemented

	def _save(self, filename):
		'''Save the parameters of a network into a file
		'''
		print('Save model in ' + filename)
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))

	def load_last(self, save_dir):
		'''Load last model from dir
		'''
		def extract_number_of_batches(filename):
			m = re.search('_nb([0-9]+)_', filename)
			return int(m.group(1))

		def extract_number_of_epochs(filename):
			m = re.search('_ne([0-9]+(\.[0-9]+)?)_', filename)
			return float(m.group(1))

		# Get all the models for this RNN
		file = save_dir + self._get_model_filename("*")
		file = np.array(glob.glob(file))

		if len(file) == 0:
			print('No previous model, starting from scratch')
			return 0

		# Find last model and load it
		last_batch = np.amax(np.array(map(extract_number_of_epochs, file)))
		last_model = save_dir + self._get_model_filename(last_batch)
		print('Starting from model ' + last_model)
		self.load(last_model)

		return last_batch


	def _load(self, filename):
		'''Load parameters values from a file
		'''


	def _input_size(self):
		''' Returns the number of input neurons
		'''
		return self.n_items

	def _get_features(self, item):
		'''Change a tuple (item_id, rating) into a list of features to feed into the RNN
		features have the following structure: [one_hot_encoding, personal_rating on a scale of ten, average_rating on a scale of ten, popularity on a log scale of ten]
		'''

		#item = (item_id, rating)

		one_hot_encoding = np.zeros(self.n_items)
		one_hot_encoding[item[0]] = 1
		return one_hot_encoding