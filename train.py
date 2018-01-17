from __future__ import print_function

import numpy as np
import helpers.command_parser as cp
import helpers.command_parser as parse
import helpers.early_stopping as EsParse
from helpers.data_handling import DataHandler

def training_command_parser(parser):
	parser.add_argument('--tshuffle', help='Shuffle sequences during training.', action='store_true')
	parser.add_argument('--extended_set', help='Use extended training set (contains first half of validation and test set).', action='store_true')
	parser.add_argument('-d', dest='dataset', help='Directory name of the dataset.', default='', type=str)
	parser.add_argument('--dir', help='Directory name to save model.', default='', type=str)
	parser.add_argument('--save', choices=['All', 'Best', 'None'], help='Policy for saving models.', default='Best')
	parser.add_argument('--metrics', help='Metrics for validation, comma separated', default='sps', type=str)
	parser.add_argument('--time_based_progress', help='Follow progress based on time rather than iterations.', action='store_true')
	parser.add_argument('--load_last_model', help='Load Last model before starting training.', action='store_true')
	#parser.add_argument('--progress', help='Progress intervals', default='2.', type=str)
	parser.add_argument('--progress', help='Progress intervals', default='5000', type=str)
	parser.add_argument('--mpi', help='Max progress intervals', default=np.inf, type=float)
	parser.add_argument('--max_iter', help='Max number of iterations', default=np.inf, type=float)
	parser.add_argument('--max_time', help='Max training time in seconds', default=np.inf, type=float)
	parser.add_argument('--min_iter', help='Min number of iterations before showing progress', default=50000, type=float) # 10 epoch for ml1m
	#parser.add_argument('--min_iter', help='Min number of iterations before showing progress', default=999,	type=float)

def num(s):
	try:
		return int(s)
	except ValueError:
		return float(s)

def main():

	args = cp.command_parser(training_command_parser, cp.predictor_command_parser, EsParse.early_stopping_command_parser)
	predictor = parse.get_predictor(args)

	dataset = DataHandler(dirname=args.dataset, extended_training_set=args.extended_set, shuffle_training=args.tshuffle)
	if args.dataset == "ml1m":
		args.min_iter = 50000
		args.progress = 5000
	elif args.dataset == 'netflix':
		args.min_iter = 600000
		args.progress = 40000
	elif args.dataset == 'rsc':
		args.min_iter = 800000
		args.progress = 200000

	predictor.prepare_model(dataset)
	predictor.train(dataset,
		save_dir=dataset.dirname + "models/" + args.dir, 
		time_based_progress=args.time_based_progress, 
		progress=num(args.progress), 
		autosave=args.save, 
		max_iter=args.max_iter,
		min_iterations=args.min_iter,
		max_time=args.max_time,
		early_stopping=EsParse.get_early_stopper(args),
		load_last_model=args.load_last_model,
		validation_metrics=args.metrics.split(','))

if __name__ == '__main__':
	main()
