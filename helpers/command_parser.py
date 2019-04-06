import argparse

from neural_networks.sequence_noise import sequence_noise_command_parser, get_sequence_noise
from neural_networks.target_selection import target_selection_command_parser, get_target_selection

def command_parser(*sub_command_parser):
	""" *sub_command_parser should be callables that will add arguments to the command parser
	"""

	parser = argparse.ArgumentParser()

	for scp in sub_command_parser:
		scp(parser)

	args = parser.parse_args()
	return args

def predictor_command_parser(parser):
	parser.add_argument('-fr', dest='framework', choices=['ktf', 'kth', 'th', 'tf'], help='framework, [keras-tensorflow backend, keras-theano backend, theano, tensorflow]', default='tf')
	parser.add_argument('-b', dest='batch_size', help='Batch size', default=16, type=int)
	parser.add_argument('-l', dest='learning_rate', help='Learning rate', default=0.1, type=float)
	parser.add_argument('-r', dest='regularization', help='Regularization (positive for L2, negative for L1)', default=0., type=float)
	parser.add_argument('--loss', help='Loss function, choose between TOP1, BPR and Blackout (Sampling), or hinge, logit and logsig (multi-targets), or CCE (Categorical cross-entropy)',
						default='CCE', type=str)
	parser.add_argument('--db', dest='diversity_bias', help='Diversity bias (for RNN with CCE, TOP1, BPR or Blackout loss)', default=0.0, type=float)
	parser.add_argument('--max_length', help='Maximum length of sequences during training (for RNNs)', default=30, type=int)
	parser.add_argument('--act', help='activation function in recurrent layer', choices=['relu', 'elu', 'lrelu', 'tanh'], default='tanh', type=str)
	parser.add_argument('--save_log', help='log history when using tensorflow', action='store_true')
	parser.add_argument('--log_dir', help='Directory name for saving tensorflow log.', default='log', type=str)
	parser.add_argument('--mem_frac', help='memory fraction for tensorflow', default=0.3, type=float)

	parser.add_argument('--tying', help='tying embedding layer', action='store_true')
	parser.add_argument('--temp', help='temperature parameter', default=10, type=int)
	parser.add_argument('--gamma', help='gamma', default=0.5, type=float)
	parser.add_argument('--iter', help='train iteratively in every user subsequences', action='store_true')
	parser.add_argument('--tying_new', help='modified tying, use same output with new target vector', action='store_true')
	parser.add_argument('--att', help='use attention mechanism', action='store_true')

	namespace, args = parser.parse_known_args() #temporary args for checking framework

	if getattr(namespace, 'framework') == 'th':
		from neural_networks.update_manager_th import update_manager_command_parser
		from neural_networks.recurrent_layers_th import recurrent_layers_command_parser
		update_manager_command_parser(parser)
		recurrent_layers_command_parser(parser)

	elif getattr(namespace, 'framework') == 'tf':
		from neural_networks.update_manager import update_manager_command_parser
		from neural_networks.recurrent_layers import recurrent_layers_command_parser
		update_manager_command_parser(parser)
		recurrent_layers_command_parser(parser)

	else:
		from neural_networks.update_manager_k import update_manager_command_parser
		from neural_networks.recurrent_layers import recurrent_layers_command_parser
		recurrent_layers_command_parser(parser)
		update_manager_command_parser(parser)

	sequence_noise_command_parser(parser)
	target_selection_command_parser(parser)

	#result = parser.parse_args()
	#print(result)

def get_predictor(args):

	if args.framework == 'th':
		from neural_networks.update_manager_th import get_update_manager
		from neural_networks.recurrent_layers_th import get_recurrent_layers
		updater = get_update_manager(args)
		recurrent_layer = get_recurrent_layers(args)
	elif args.framework == 'tf':
		from neural_networks.update_manager import get_update_manager
		from neural_networks.recurrent_layers import get_recurrent_layers
		updater = get_update_manager(args)
		recurrent_layer = get_recurrent_layers(args)
	else:
		from neural_networks.update_manager_k import get_update_manager
		from neural_networks.recurrent_layers import get_recurrent_layers
		updater = get_update_manager(args)
		recurrent_layer = get_recurrent_layers(args)

	sequence_noise = get_sequence_noise(args)
	target_selection = get_target_selection(args)

	if args.framework == 'th':
		from neural_networks.rnn_onehot_theano import RNNOneHotTH

		return RNNOneHotTH(max_length=args.max_length, regularization=args.regularization, updater=updater, target_selection=target_selection,
							 sequence_noise=sequence_noise, recurrent_layer=recurrent_layer, batch_size=args.batch_size, active_f=args.act,
							tying=args.tying, temperature=args.temp, gamma=args.gamma, iter=args.iter, tying_new=args.tying_new, attention=args.att)

	elif args.framework == 'tf':
		from neural_networks.rnn_one_hot import RNNOneHotTF

		return RNNOneHotTF(mem_frac=args.mem_frac, save_log=args.save_log, log_dir=args.log_dir, max_length=args.max_length, regularization=args.regularization, updater=updater, target_selection=target_selection,
							 sequence_noise=sequence_noise, recurrent_layer=recurrent_layer, batch_size=args.batch_size, active_f=args.act,
							tying=args.tying, temperature=args.temp, gamma=args.gamma, iter=args.iter, tying_new=args.tying_new, attention=args.att)

	if args.framework == 'kth' or args.framework == 'ktf':
		from neural_networks.rnn_oh_keras import RNNOneHotK

		if args.framework == 'kth':
			backend = 'theano'
		else:
			backend = 'tensorflow'

		return RNNOneHotK(mem_frac=args.mem_frac, backend=backend, max_length=args.max_length, regularization=args.regularization, updater=updater, target_selection=target_selection,
							 sequence_noise=sequence_noise, recurrent_layer=recurrent_layer, batch_size=args.batch_size, active_f=args.act,
							tying=args.tying, temperature=args.temp, gamma=args.gamma, iter=args.iter, tying_new=args.tying_new, attention=args.att)





	