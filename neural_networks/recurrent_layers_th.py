from __future__ import print_function

import lasagne
import theano.tensor as T

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
			self.name += "b"+self.layer_type+"_"
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


	def __call__(self, input_layer, mask_layer, input_shape=None, activate_f=None):
		if self.embedding_size > 0:
			in_int32 = lasagne.layers.ExpressionLayer(input_layer, lambda x: x.astype('int32'))  # change type of input
			in_int32_idx = lasagne.layers.ExpressionLayer(in_int32, lambda x: T.argmax(x, axis=2), output_shape=(input_shape[0], input_shape[1],))
			l_emb = lasagne.layers.EmbeddingLayer(in_int32_idx, input_size=input_shape[2], output_size=self.embedding_size)
			l_rec = self.get_recurrent_layers(l_emb, mask_layer)
		else:
			l_rec = self.get_recurrent_layers(input_layer, mask_layer, activate_f=activate_f)

		return l_rec

	def get_recurrent_layers(self, input_layer, mask_layer, activate_f=lasagne.nonlinearities.tanh, only_return_final=True):

		orf = False
		prev_layer = input_layer
		for i, h in enumerate(self.layers):
			if i == len(self.layers) - 1:
				orf = only_return_final
			prev_layer = self.get_one_layer(prev_layer, mask_layer, h, activate_f, orf)

		return prev_layer

	def get_one_layer(self, input_layer, mask_layer, n_hidden, activate_f, only_return_final):
		if self.bidirectional:
			forward = self.get_unidirectional_layer(input_layer, mask_layer, n_hidden, activate_f, only_return_final, backwards=False)
			backward = self.get_unidirectional_layer(input_layer, mask_layer, n_hidden, activate_f, only_return_final, backwards=True)
			# return lasagne.layers.ConcatLayer([forward, backward], axis = -1)
			return lasagne.layers.ElemwiseSumLayer([forward, backward])
		else:
			return self.get_unidirectional_layer(input_layer, mask_layer, n_hidden, activate_f, only_return_final, backwards=False)

	def get_unidirectional_layer(self, input_layer, mask_layer, n_hidden, activate_f, only_return_final, backwards=False):
		if self.layer_type == "LSTM":
			layer = lasagne.layers.LSTMLayer
		elif self.layer_type == "GRU":
			layer = lasagne.layers.GRULayer
		elif self.layer_type == "Vanilla":
			layer = lasagne.layers.RecurrentLayer
		else:
			raise ValueError('Unknown layer type')

		return layer(input_layer, n_hidden, mask_input=mask_layer, grad_clipping=self.grad_clip,
					 learn_init=True, nonlinearity=activate_f, only_return_final=only_return_final, backwards=backwards)