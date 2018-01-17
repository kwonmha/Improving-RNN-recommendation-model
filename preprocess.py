# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import pandas as pd
import argparse
import os
import sys
from shutil import copyfile

def command_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='filename', help='Input file', default='C:/Users\Kwon\workspace\Python\RNN-Rec\data\ml1m.dat', type=str)
	parser.add_argument('--columns', help='Order of the columns in the file (eg: "uirt"), u for user, i for item, t for timestamp, r for rating. '
										  'If r is not present a default rating of 1 is given to all interaction. If t is not present interactions are assumed to be in chronological order. Extra columns are ignored. Default: uit', default="uirt", type=str)
	parser.add_argument('--sep', help='Separator between the column. If unspecified pandas will try to guess the separator', default="::", type=str)
	parser.add_argument('--min_user_activity', help='Users with less interactions than this will be removed from the dataset. Default: 2', default=2, type=int)
	parser.add_argument('--min_item_pop', help='Items with less interactions than this will be removed from the dataset. Default: 5', default=5, type=int)
	parser.add_argument('--val_size', help='Number of users to put in the validation set. If in (0,1) it will be interpreted as the fraction of total number of users. Default: 0.1', default=500, type=float)
	parser.add_argument('--test_size', help='Number of users to put in the test set. If in (0,1) it will be interpreted as the fraction of total number of users. Default: 0.1', default=500, type=float)
	parser.add_argument('--seed', help='Seed for the random train/val/test split', default=2, type=int)
	parser.add_argument('--min_time_diff', help='avoid users whose ratingTime is too dense', default=0, type=int)
	parser.add_argument('--w2v_dim', help='dim of word vector', default=0, type=int)
	parser.add_argument('--w2v_win', help='window size for training word2vec', default=10, type=int)
	parser.add_argument('--sg', help='decide w2v architecture, 1 for sg, 0 for cbow, default is 0', default=0, type=int)
	parser.add_argument('--w2v_iter', help='iteration of w2v training', default=1000, type=int)
	parser.add_argument('--ow', help='train w2v only', action='store_true')

	args = parser.parse_args()
	args.dirname = os.path.splitext(args.filename)[0] + "/" # make dir for diff dataset
	return args

def warn_user(dirname):
	"""Ask user if he's sure to create files in that directory.
	"""
	print('This program will create a lot of files and directories in ' + dirname)
	answer = input('Are you sure that you want to do that ? [y/n]')
	if answer != "y":
		sys.exit(0)

def create_dirs(dirname):
	if not os.path.exists(dirname + "data"):
		os.makedirs(dirname + "data")

	if not os.path.exists(dirname + "models"):
		os.makedirs(dirname + "models")

	if not os.path.exists(dirname + "results"):
		os.makedirs(dirname + "results")

def load_data(filename, columns, separator):
	""" Load the data from filename and sort it according to timestamp.
	Returns a dataframe with 3 columns: user_id, item_id, rating
	"""

	print('Load data...')
	# colums = ex) "uirt" (user, item, rating, time)
	# Pandas DataFrame type
	data = pd.read_csv(filename, sep=separator, names=list(columns), index_col=False, usecols=range(len(columns)))

	# If r is not present a default rating of 1 is given to all interaction.
	if 'r' not in columns:
		# Add a column of default ratings
		data['r'] = 1
	#print(data)
	#index	u		i		r		t
	#0,1~	uid		movieid	1~5		timestamp
	# 무조건 column 이름을, 데이터와 상관없이 arg에서 입력한 순서대로 가져온다.
	if 't' in columns:
		# sort according to the timestamp column
		if data['t'].dtype == np.int64: # probably a timestamp
			data['t_s'] = data['t'].copy()
			data['t'] = pd.to_datetime(data['t_s'], unit='s')
			#print("type == np.int64") 여기로 보통 들어오는 듯
		else: # type == 날짜 시간
			"""
			print(data['t'].dtype)
			data['t_s'] = data['t'].copy()
			print(data['t_s'].head(10))
			data['t'] = pd.to_datetime(data['t_s'], unit='s')
			#print("type != np.int64")
			"""
			data['t'] = pd.to_datetime(data['t'])
		print('Sort data in chronological order...')
		data.sort_values('t', inplace=True)
		data['t'] = data.t.dt.date
		#print(type(data['t'][0])) datetime.date
		#print(data['date'].dt.date)

	return data

def remove_rare_elements(data, min_user_activity, min_item_popularity, min_time_difference):
	"""Removes user and items that appears in too few interactions.
	min_user_activity is the minimum number of interaction that a user should have.
	min_item_popularity is the minimum number of interaction that an item should have.
	NB: the constraint on item might not be strictly satisfied because rare users and items are removed in alternance, 
	and the last removal of inactive users might create new rare items.

	!!!!!!!!!!!
	Removes users who rated all itmes in min_time_difference days
	"""
	
	print('Remove inactive users and rare items...')

	#print(data.u.nunique())
	#Remove inactive users a first time
	user_activity = data.groupby('u').size()
	#print(user_activity)
	#print(np.in1d(data.u, user_activity[user_activity >= min_user_activity].index))
	data = data[	np.in1d(data.u, user_activity[user_activity >= min_user_activity].index)	]

	#Remove unpopular items
	item_popularity = data.groupby('i').size()
	data = data[np.in1d(data.i, item_popularity[item_popularity >= min_item_popularity].index)]

	#Remove users that might have passed below the activity threshold due to the removal of rare items
	#꼭 해야 하나?
	user_activity = data.groupby('u').size()
	data = data[np.in1d(data.u, user_activity[user_activity >= min_user_activity].index)]

	# #Remove items rated too densly regarding to timestamp
	# time_diff = data.groupby('u')['t'].max() - data.groupby('u')['t'].min()
	# #print(time_diff[time_diff >= datetime.timedelta(days=min_time_difference)].index)
	# data = data[np.in1d(data.u, time_diff[time_diff >= datetime.timedelta(days=min_time_difference)].index)]

	# 날짜가 너무 이전인 영화들을 지워본다. time gap이 너무 클 경우. 현재 0.01 셋 : 97.3.14 <-> 99.12..31 이다.

	return data

def save_index_mapping(data, dirname):
	""" Save the mapping of original user and item ids to numerical consecutive ids in dirname.
	NB: some users and items might have been removed in previous steps and will therefore not appear in the mapping.
	"""
	separator = "\t"

	# Pandas categorical type will create the numerical ids we want
	print('Map original users and items ids to consecutive numerical ids...')

	data['u_original'] = data['u'].astype('category')
	# print(data['u'])
	# print("=========================================================================================")
	# print(data['u_original'])
	data['u'] = data['u_original'].cat.codes
	# print("=========================================================================================")
	# print(data['u'])

	data['i_original'] = data['i'].astype('category')
	data['i'] = data['i_original'].cat.codes 	#코드화. ex sunny windy .. -> 0 1 2..

	print('Save ids mapping to file...')
	user_mapping = pd.DataFrame({'original_id' : data['u_original'], 'new_id': data['u']})
	# print(user_mapping)
	# print("=========================================================================================")
	user_mapping.sort_values('original_id', inplace=True) #원래 시간 순에서 다시 original_id 순으로?
	#print(user_mapping)
	user_mapping.drop_duplicates(subset='original_id', inplace=True)
	#print(user_mapping)
	user_mapping.to_csv(dirname+"data/user_id_mapping", sep=separator, index=False)

	item_mapping = pd.DataFrame({'original_id' : data['i_original'], 'new_id': data['i']})
	item_mapping.sort_values('original_id', inplace=True)
	item_mapping.drop_duplicates(subset='original_id', inplace=True)
	item_mapping.to_csv(dirname+"data/item_id_mapping", sep=separator, index=False)

	return data

def split_data(data, nb_val_users, nb_test_users, dirname, save_time):
	"""Splits the data set into training, validation and test sets.
	Each user is in one and only one set.
	nb_val_users is the number of users to put in the validation set.
	nb_test_users is the number of users to put in the test set.
	"""
	nb_users = data['u'].nunique()

	# check if nb_val_user is specified as a fraction
	if nb_val_users < 1:
		nb_val_users = round(nb_val_users * nb_users)
	if nb_test_users < 1:
		nb_test_users = round(nb_test_users * nb_users)
	nb_test_users = int(nb_test_users)
	nb_val_users = int(nb_val_users)

	if nb_users <= nb_val_users+nb_test_users:
		raise ValueError('Not enough users in the dataset: choose less users for validation and test splits')

	def extract_n_users(df, n):
		users_ids = np.random.choice(df['u'].unique(), n, replace=False)
		n_set = df[df['u'].isin(users_ids)]
		remain_set = df.drop(n_set.index)
		return n_set, remain_set

	print('Split data into training, validation and test sets...')
	test_set, tmp_set = extract_n_users(data, nb_test_users)
	val_set, train_set = extract_n_users(tmp_set, nb_val_users)


	print('Save training, validation and test sets in the triplets format...')

	if save_time:
		train_set.to_csv(dirname + "data/train_set_triplets", sep="\t", columns=['u', 'i', 'r', 't'], index=False, header=False)
		val_set.to_csv(dirname + "data/val_set_triplets", sep="\t", columns=['u', 'i', 'r', 't'], index=False, header=False)
		test_set.to_csv(dirname + "data/test_set_triplets", sep="\t", columns=['u', 'i', 'r', 't'], index=False, header=False)
	else:
		train_set.to_csv(dirname + "data/train_set_triplets", sep="\t", columns=['u', 'i', 'r'], index=False,
						 header=False)
		val_set.to_csv(dirname + "data/val_set_triplets", sep="\t", columns=['u', 'i', 'r'], index=False, header=False)
		test_set.to_csv(dirname + "data/test_set_triplets", sep="\t", columns=['u', 'i', 'r'], index=False,
						header=False)

	return train_set, val_set, test_set

def gen_sequences(data, half=False):
	"""Generates sequences of user actions from data.
	each sequence has the format [user_id, first_item_id, first_item_rating, first_item_rating_time, 2nd_item_id, 2nd_item_rating, 2nd_item_rating_time...].
	If half is True, cut the sequences to half their true length (useful to produce the extended training set).
	half for sequence+ file.
	"""
	data = data.sort_values('u', kind="mergesort") # Mergesort is stable and keeps the time ordering
	seq = []
	prev_id = -1
	for u, i, r, t in zip(data['u'], data['i'], data['r'], data['t']):
		if u != prev_id:
			if len(seq) > 3:
				if half:
					seq = seq[:1+2*int((len(seq) - 1)/4)]
				yield seq
			prev_id = u
			seq = [u]
		seq.extend([i, r, t])
	if half:
		seq = seq[:1+2*int((len(seq) - 1)/4)]
	yield seq

def w2v_generator(data):
	"""data의 모든 user에 대해 모든 item의 리스트를 yiled"""
	data = data.sort_values('u', kind="mergesort") # Mergesort is stable and keeps the time ordering
	#print(data)
	#data = data(epochs=1)
	seq = []
	prev_id = -1
	for u, i in zip(data['u'], data['i']):
		#print(u, i)
		if u != prev_id:
			if len(seq) >= 1:
				# print(prev_id)
				# print(seq)
				# exit()
				yield seq
			prev_id = u
			seq = [str(i)]
		else:
			seq.extend([str(i)])
	#print(type(i))

	yield seq

def save_word2vec(data, args):
	print("save word2vec model")
	w2v = word2vec.Word2Vec(iter=args.w2v_iter, min_count=1, size=args.w2v_dim, window=args.w2v_win, sg=args.sg)
	#print(data['i'].nunique())
	w2v.build_vocab([map(str, range(data['i'].nunique()))])
	w2v.train(w2v_generator(data))
	import w2v_test
	w2v_test.showTitles(w2v.similar_by_vector(w2v['127']))
	answer = raw_input('Save model ? [y/n]')
	if answer != "y":
		sys.exit(0)
	w2v.save(args.dirname + "w2v_d" + str(args.w2v_dim) + "_w" + str(args.w2v_win) + "_i" + str(args.w2v_iter) + "_sg" + str(args.sg))

def make_sequence_format(train_set, val_set, test_set, dirname):
	"""Convert the train/validation/test sets in the sequence format and save them.
	Also create the extended training sequences, which countains the first half of the sequences of users in the validation and test sets.
	"""

	print('Save the training set in the sequences format...')
	with open(dirname+"data/train_set_sequences", "w") as f:
		for s in gen_sequences(train_set):
			f.write(' '.join(map(str, s)) + "\n")

	print('Save the validation set in the sequences format...')
	with open(dirname+"data/val_set_sequences", "w") as f:
		for s in gen_sequences(val_set):
			f.write(' '.join(map(str, s)) + "\n")

	print('Save the test set in the sequences format...')
	with open(dirname+"data/test_set_sequences", "w") as f:
		for s in gen_sequences(test_set):
			f.write(' '.join(map(str, s)) + "\n")

	# sequences+ contains all the sequences of train_set_sequences plus half the sequences of val and test sets
	print('Save the extended training set in the sequences format...')
	copyfile(dirname+"data/train_set_sequences", dirname+"data/train_set_sequences+")
	with open(dirname+"data/train_set_sequences+", "a") as f:
		for s in gen_sequences(val_set, half=True):
			f.write(' '.join(map(str, s)) + "\n")
		for s in gen_sequences(test_set, half=True):
			f.write(' '.join(map(str, s)) + "\n")


def save_data_stats(data, train_set, val_set, test_set, dirname):
	print('Save stats...')
	# pd.set_option('display.max_rows', len(data))
	#np.set_printoptions(threshold=np.inf)
	#print(data['u'].unique())

	def _get_stats(df):
		#print(type(df.groupby('u').size())) #Series
		#print(type(df.groupby('u')['t_s'].max() - df.groupby('u')['t_s'].min())) # Series of timedelta64[ns]
		#print(pd.to_datetime(df.groupby('u')['t_s'].max() - df.groupby('u')['t_s'].min(), unit='s') - pd.to_datetime(0, unit='s'))  # Series of timedelta64[ns]
		#df['i/t'] = df.groupby('u').size() * 60 * 60 / (df.groupby('u')['t_s'].max() - df.groupby('u')['t_s'].min())
		#df['t/i'] = (df.groupby('u')['t'].max() - df.groupby('u')['t'].min()) / df.groupby('u').size()
		#print(min(df['t']))
		return "\t".join(map(str, [df['u'].nunique(), df['i'].nunique(), len(df.index), df.groupby('u').size().max(), min(df['t']).strftime("%Y-%m-%d"), max(df['t']).strftime("%Y-%m-%d")]))

	def _get_info(df):
		return "\n".join(map(str, zip(df.sort_values('u')['u'].unique(), df.groupby('u')['t'].max() - df.groupby('u')['t'].min(), df.groupby('u').size())))

	with open(dirname+"data/report", "w") as f:
		f.write("userId\t\tmax_time - min_time\t\tratedItems\n")
		f.write(_get_info(data) + "\n")

	with open(dirname+"data/stats", "w") as f:
		f.write("set\tn_users\tn_items\tn_interactions\tlongest_sequence\tearliest_time\tlatest_time\n")
		f.write("Full\t"+ _get_stats(data) + "\n") 
		f.write("Train\t"+ _get_stats(train_set) + "\n") 
		f.write("Val\t"+ _get_stats(val_set) + "\n") 
		f.write("Test\t"+ _get_stats(test_set) + "\n")

	pd.reset_option('display.max_rows')

def make_readme(dirname, val_set, test_set, min_time_diff):
	data_readme = """The following files were automatically generated by preprocess.py

	user_id_mapping
		mapping between the users ids in the original dataset and the new users ids.
		the first column contains the new id and the second the original id.
		Inactive users might have been deleted from the original, and they will therefore not appear in the id mapping.

	item_id_mapping
		Idem for item ids.

	train_set_triplets
		Training set in the triplets format.
		Each line is a user item interaction in the form (user_id, item_id, rating). 
		Interactions are listed in chronological order.

	train_set_sequences
		Training set in the sequence format.
		Each line contains all the interactions of a user in the form (user_id, first_item_id, first_rating, 2nd_item_id, 2nd_rating, ...).

	train_set_sequences+
		Extended training set in the sequence format.
		The extended training set contains all the training set plus the first half of the interactions of each users in the validation and testing set.

	val_set_triplets
		Validation set in the triplets format

	val_set_sequences
		Validation set in the sequence format

	test_set_triplets
		Test set in the triplets format

	test_set_sequences
		Test set in the sequence format

	stats
		Contains some informations about the dataset.

	The training, validation and test sets are obtain by randomly partitioning the users and all their interactions into 3 sets.
	The validation set contains {n_val} users, the test_set {n_test} users and the train set all the other users.

	""".format(n_val=str(val_set['u'].nunique()), n_test=str(test_set['u'].nunique()))

	if min_time_diff :
		data_readme += """
		min time diff = {min_time_diff} days
		""".format(min_time_diff=min_time_diff)

	results_readme = """The format of the results file is the following
	Each line correspond to one model, with the fields being:
		Number of epochs
		precision
		sps
		user coverage
		number of unique items in the test set
		number of unique items in the recommendations
		number of unique items in the succesful recommendations
		number of unique items in the short-term test set (when the goal is to predict precisely the next item)
		number of unique items in the successful short-term recommendations
		recall
		NDCG
	NB: all the metrics are computed "@10"
	"""

	with open(dirname+"data/README", "w") as f: 
		f.write(data_readme)
	with open(dirname+"results/README", "w") as f: 
		f.write(results_readme)

def main():
	
	args = command_parser()
	rand = np.random.seed(seed=args.seed)
	#warn_user(args.dirname)
	create_dirs(args.dirname)
	data = load_data(args.filename, args.columns, args.sep)
	#data = remove_rare_elements(data, args.min_user_activity, args.min_item_pop, args.min_time_diff)
		#check seq length info
		# print(max(data.groupby('u').size()))
		# print(min(data.groupby('u').size()))
		# print(np.mean(np.array(data.groupby('u').size())))
		# exit()
	data = save_index_mapping(data, args.dirname)
	if args.w2v_dim:
		save_word2vec(data, args)
		print("w2v training done")
	if not args.ow:
		train_set, val_set, test_set = split_data(data, args.val_size, args.test_size, args.dirname, args.min_time_diff)
		make_sequence_format(train_set, val_set, test_set, args.dirname)
		save_data_stats(data, train_set, val_set, test_set, args.dirname)
		make_readme(args.dirname, val_set, test_set, args.min_time_diff)

		print('Data ready!')

		print(data.head(10))
		# print(data.groupby('u'))
		# print(data.iloc[2, 3])
		# print(data.iloc[3, 3] - data.iloc[1, 3])
		# print(type(data.iloc[3, 3] - data.iloc[1, 3]))
		# print((data.iloc[3, 3] - data.iloc[1, 3]) < datetime.timedelta(minutes=1))

if __name__ == '__main__':
	main()