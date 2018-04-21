from __future__ import print_function

import tensorflow as tf

def recurrent_layers_command_parser(parser):
	parser.add_argument('--r_t', dest='recurrent_layer_type', choices=['LSTM', 'GRU', 'Vanilla'], help='Type of recurrent layer', default='LSTM')
	parser.add_argument('--r_l', help="Layers' size, (eg: 100-50-50)", default="50", type=str)
	parser.add_argument('--r_bi', help='Bidirectional layers.', action='store_true')
	parser.add_argument('--r_emb',
						help='Add an embedding layer before the RNN. Takes the size of the embedding as parameter, a size<1 means no embedding layer.',
						type=int, default=0)
	parser.add_argument('--ntd', help='do not get distribution for target, only tying', action='store_true')
	parser.add_argument('--nwd', help='only get distribution for target(do not tying)', action='store_true')

def get_recurrent_layers(args):
	return RecurrentLayers(layer_type=args.recurrent_layer_type, layers=list(map(int, args.r_l.split('-'))), bidirectional=args.r_bi,
						   embedding_size=args.r_emb, ntd=args.ntd, nwd=args.nwd)

class RecurrentLayers(object):
	def __init__(self, layer_type="LSTM", layers=[32], bidirectional=False, embedding_size=0, grad_clipping=100, ntd=False, nwd=False):
		super(RecurrentLayers, self).__init__()
		self.layer_type = layer_type
		self.layers = layers
		self.bidirectional = bidirectional
		self.embedding_size = embedding_size
		self.grad_clip=grad_clipping
		self.no_td = ntd
		self.no_wt = nwd
		self.set_name()

	def set_name(self):

		self.name = ""
		if self.bidirectional:
			self.name += "bi"+self.layer_type+"_"
		elif self.layer_type != "LSTM":
			self.name += self.layer_type+"_"

		self.name += "gc"+str(self.grad_clip)+"_"
		if self.embedding_size > 0:
			self.name += "e"+str(self.embedding_size) + "_"
			if self.no_td :
				self.name += "ntd_"
			if self.no_wt:
				self.name += "nwd_"
		self.name += "h"+('-'.join(map(str,self.layers)))


	def __call__(self, input, seq_len=None, activate_f=None):

		if self.layer_type == "LSTM":
			rnn_cell = tf.nn.rnn_cell.LSTMCell
			cells = [rnn_cell(h, activation=activate_f, state_is_tuple=True) for _, h in enumerate(self.layers)]
		elif self.layer_type == "GRU":
			rnn_cell = tf.nn.rnn_cell.GRUCell
			cells = [rnn_cell(h, activation=activate_f) for _, h in enumerate(self.layers)]
		elif self.layer_type == "Vanilla":
			rnn_cell = tf.nn.rnn_cell.RNNCell
			cells = [rnn_cell(h, activation=activate_f, state_is_tuple=True) for _, h in enumerate(self.layers)]
		else:
			raise ValueError('Unknown layer type')


		multi_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
		if self.bidirectional:
			rnn_outputs, _states = tf.nn.bidirectional_dynamic_rnn(multi_cells, multi_cells, input, sequence_length=seq_len, dtype=tf.float32)
		else:
			rnn_outputs, _states = tf.nn.dynamic_rnn(multi_cells, input, sequence_length=seq_len, dtype=tf.float32)
			# last_output, rnn_outputs, _states = my_rnn.dynamic_rnn(multi_cells, input, return_sequences=False, sequence_length=seq_len, dtype=tf.float32)
			# rnn_outputs, _states = my_rnn.dynamic_rnn(multi_cells, input, sequence_length=seq_len, dtype=tf.float32)

		return rnn_outputs, _states

