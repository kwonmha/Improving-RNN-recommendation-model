# Improving RNN based recommendtation
Tensorflow, Keras implementation of "[Collaborative filtering based on sequences](https://github.com/rdevooght/sequence-based-recommendations)". 
Also, theano is available in this project. 
All methods other than RNN-Categorical cross entropy are removed to focus on improving the performance of RNN and for simplification. 
I applied "[weight tying technique](https://pdfs.semanticscholar.org/fb26/4cfa7309c572dc1aa6e70e745367f97ef78e.pdf)" into the RNN-CCE model and it resulted it higher recall in MovieLens 1M dataset.
This is only available in tensorflow now.

For reproducibility, you can download the same preprocessed data sequence from http://iridia.ulb.ac.be/~rdevooght/rnn_cf_data.zip. 
<p align="center">
	<img width="1000" height="400" src="https://user-images.githubusercontent.com/8953934/40874985-dc9ae5fe-66b2-11e8-8d50-3339bbfc5938.png">
</p>
<p align="center">
	<img width="1000" height="400" src="https://user-images.githubusercontent.com/8953934/40876662-8bb4478c-66b4-11e8-9e00-c03a922d5f50.JPG">
</p>

## Experimental results (Performed on Tensorflow)
| Models                      | sps                 | Recall               |
|-----------------------------|:-------------------:|:--------------------:|
| LSTM_vanilla(dim:50)        | 31                  | 6.2                  |
| LSTM_weight_tying(dim:50)   | 29.2                | 6.0                  |
| LSTM_vanilla(dim:100)       | 29.2                | 5.2                  |
| LSTM_weight_tying(dim:100)  | 33.8                | 6.6                  |
| LSTM_vanilla(dim:200)       | 28.6                | 5.6                  |
| LSTM_weight_tying(dim:200)  | 31                  | 6.8                  |

**Weight tying might not that effective to improve sps but it clearly marks higher recall.
And its effectiveness starts to come out when the dimension of the hidden state is higher than 50.**

## Requirements

- Python 3
- Tensorflow >= 1.4.0 or keras >=2.1.2 or theano 0.8.2 with lasagne 0.2.dev1
- Numpy
- _pickle
- h5py for saving keras model

## Installation
This project runs on windows 10, 64bit. You can use any deeplearning framework among tensorflow, keras and theano. I used anaconda containing all of them. 

You can use
````
pip install tensorflow-gpu
````

And maybe mkl or numpy-mkl library is needed to run tensorflow. Also it is known that the development of theano has been stopped so I personally recommend to use tensorflow or keras with tensorflow backend.


## Simple Report
It takes much more time to run with keras-either tensorflow backend or theano backend. You'd better to use raw tensorflow or theano. 

The results made by rdevooght's paper seems only available in python 2. It was unable to reproduce in python3 although I copied his theano code based on python2 and modified to fit python3. But the result is reproducible in python2. The performance of models in this project is little bit lower than that of the original project.

**When I tested with theano in python3, only Adadelta and Adam optimizer didn't make nan as loss. But Adadelta was slower than Adam, so the only good option for theano would be Adam.**


## Usage

Use `-fr` option to select deeplearning framework. **'tf'** for tensorflow, **'ktf'** for keras-tensorflow backend, **'kth'** for keras-theano backend, **'th'** for theano. Available on train.py and test.py.

Example : `python train.py -fr tf`

`--mem_frac` sets the fraction of memory to use for tensorflow or keras-tensorflow backend.
`--save_log` You can set whether to save training logs when using tensorflow.
`--log_dir` sets the directory to save training logs when using tensorflow.

**Explanation below are almost copied from original github and simplified**

### train.py

This script is used to train models and offers many options regarding when to save new models and when to stop training.
The basic usage is the following:
````
python train.py -d path/to/dataset/ 
````

The argument `-d` is used to specify the path to the folder that contains the "data", "models" and "results" subfolders created by preprocess.py. 
If you have multiple datasets with a partly common path (e.g. path/to/dataset1/, path/to/dataset2/, etc.) you can specify this common path in the variable DEFAULT_DIR of helpers/data_handling.py. For example, setting DEFAULT_DIR = "path/to/" and using the argument `-d dataset1` will look for the dataset in "path/to/dataset1/".

The optional arguments are the following:

Option | Desciption
------ | ----------
`--dir dirname/` | Name of the subfolder of "path/to/dataset/models/" in which to save the model. By default it will be saved directly in the models/ folder, but using subfolders can be useful when many models are tested.
`--progress {int or float}` | Number of iterations (or seconds) between two evaluations of the model on the validation set. When the model is evaluated, progress is shown on the command line, and the model might be saved (depending on the `--save` option). An float value means that the evaluations happen at geometric intervals (rather than linear). Default: 5000
`--metrics value` | Metrics computed on the validation set, separated by commas. Available metrics are recall, sps, ndcg, item\_coverage, user\_coverage and blockbuster\_share. Default: sps.
`--save [All, Best, None]` | Policy for saving models. If "None", no model is saved. If "All", the current model is saved each time the model is evaluated on the validation set, and no model is destroyed. If "Best", the current model is only saved if it improves over the previous best results on the validation set, and the previous best model is deleted. If "Best" and multiple metrics are used, all the pareto-optimal models are saved. 
`--time_based_progress` | Base the interval between two evaluations on the number of elapsed seconds rather than on the number of iterations.
`--mpi value` | Max number of iterations (or seconds) between two evaluations (useful when using geometric intervals). Default: inf.
`--max_iter value` | Max number of iterations (default: inf).
`--max_time value` | Max training time in seconds (default: inf).
`--min_iter value` | Min number of iterations before making the first evaluation (default: 0).
`--extended_set` | Use extended training set (contains first half of validation and test set). This is necessary for factorization based methods such as BPRMF and FPMC because they need to build a model for every user.
`--tshuffle` | Shuffle the order of sequences between epochs.
`--load_last_model` | Load Last model before starting training (it will search for a model build with all the same options and take the one with the largest number of epochs).
`--es_m [WorstTimesX, StopAfterN, None]` | Early stopping method (by default none is used, and training continues until max_iter or max_time is reached). WorstTimesX will stop training if the number of iterations since the last best score on the validation set is longer than X times the longest time between two consecutive best scores. StopAfterN will stop the training if the model has not improved for the N last evaluations on the validation set.
`--es_n N` | N parameter for StopAfterN (default: 5).
`--es_x X` | X parameter for WorstTimesX (default: 2).
`--es_min_wait num_epochs` | Mininum number of epochs before stopping (for WorstTimesX). Default: 1.
`--es_LiB` | Lower is better for validation score. By default a higher validation score is considered better, but if it is not the case you can use this option.

The options specific to each method are explained in the Methods section.

### test.py

This script test the models built with train.py on the test set.
The basic usage is:
````
python test.py -d path/to/dataset/ -m Method_name
````
The argument `-d` works in the same way as with train.py, and the precise model to test is specified by the `--dir` option and the methods-specific options.
If multiple models fit the options (They are in the same subfolder and were trained with the same method and same options), they are all evaluated one after the other, except if the argument `-i epoch_number` is also specified, which will then select the model based on the number of epochs.

`--metrics` allows to specify the list of metrics to compute, separated by commas. By default the metrics are: sps, recall, item\_coverage, user\_coverage, blockbuster_share.
The "blockbuster share" is the percentage of correct recommendations among the 1% most popular items.
The other available metrics are the sps, the ndcg and the assr (when clustering is used).

All the metrics are computed "@k", with k=10 by default. k can be changed using the `-k` option.

When the `--save` option is used, the results are saved in a file in "path/to/dataset/results/".
the results of each model form a line of the file, and each line contains the number of epochs followed by the metrics specified by `--metrics`.

`-iter` In default(when this option is False) test.py make one input sequence per each users in test data for testing model dividing movie history into half. 
If a user watched n movies, use m_1, m_2 ... m_n/2 as input and the others as criteria.
If this is set True, test.py generates n input-criteria pairs like [m_1 / m_2 .. m_n], [m_1, m_2 / m_3 ... m_n] ... [m_1, m_2 .. m_n-1 / m_n].
So you can evaluate the performance of model more precisly. Of course, this takes much more time.


### Recurrent Neural Networks parameters


The RNN have many options allowing to change the type/size/number of layers, the training procedure and the objective function, and some options are specific to a particular objective function.

##### Layers

Option | Desciption
------ | ----------
`--r_t [LSTM, GRU, Vanilla]` | Type of recurrent layer (default is LSTM)
`--r_l size_of_layer1-size_of_layer2-etc.` | Size and number of layers. for example, `--r_l 100-50-50` creates a layer with 50 hidden neurons on top of another layer with 50 hidden neurons on top of a layer with 100 hidden neurons. Default: 32.
`--r_bi` | Use bidirectional layers.
`--r_emb size` | Adds an embedding layer before the recurrent layer. By default no embedding layer is used, but it is adviced to use one (e.g. `--r_emb 100`).
`--act` | Activation function for rnn.

other parameters for tying matrix are not added yet.

##### Update mechanism

Option | Desciption
------ | ----------
`--u_m [adagrad, adadelta, rmsprop, nesterov, adam]` | Update mechanism.
`--u_l float` | Learning rate (default: 0.001). The default learning rate works well with adam. For adagrad `--u_l 0.1` is adviced.
`--u_rho float` | rho parameter for Adadelta and RMSProp, or momentum for Nesterov momentum (default: 0.9).
`--u_b1 float` | Beta 1 parameter for Adam (default: 0.9).
`--u_b2 float` | Beta 2 parameter for Adam (default: 0.999).

##### Noise

Option | Desciption
------ | ----------
`--n_dropout P` | Dropout probability (default: 0.)
`--n_shuf P` | Probability that an item is swapped with another one (default: 0.).
`--n_shuf_std STD` | If an item is swapped, the position of the other item is drawn from a normal distribution whose std is defined by this parameter (default: 5.).

##### Other options

Option | Desciption
------ | ----------
`-b int` | Size of the mini-batchs (default: 16)
`--max_length int` | Maximum length of sequences (default: 30)


##### Objective functions

Option | Desciption
------ | ----------
`--loss [CCE, KL]` | Objective function. CCE is the categorical cross-entropy, KL is the KL divergence. Default is CCE.
`-r float` | *Only for CCE*. Add a regularization term. A positive value will use L2 regularization and a negative value will use L1(not added yet). Default: 0.

### preprocess.py
**If you want to make same result as the paper, just download the data from above link instead of running preprocess.py in the project.**

This script takes a file containing a dataset of user/item interactions and split it into training/validation/test sets and save them in the format used by train.py and test.py.
The original dataset must be in a format where each line correspond to a single user/item interaction.

The only required argument is `-f path/to/dataset`, which is used to specify the original dataset. The script will create subfolders named "data", "models" and "results" in the folder containing the original dataset. "data" is used by preprocess.py to store all the files it produces, "models" is used by train.py to store the trained models and "results" is used by test.py to store the results of the tests.

The optional arguments are the following:

Option | Desciption
------ | ----------
`--columns` | Order of the columns in the file (eg: "uirt"), u for user, i for item, t for timestamp, r for rating. If r is not present a default rating of 1 is given to all interaction. If t is not present interactions are assumed to be in chronological order. Extra columns are ignored. Default: uit
`--sep` | Separator between the column. If unspecified pandas will try to guess the separator
`--min_user_activity` | Users with less interactions than this will be removed from the dataset. Default: 2
`--min_item_pop` | Items with less interactions than this will be removed from the dataset. Default: 5
`--val_size` | Number of users to put in the validation set. If in (0,1) it will be interpreted as the fraction of total number of users. Default: 0.1
`--test_size` | Number of users to put in the test set. If in (0,1) it will be interpreted as the fraction of total number of users. Default: 0.1
`--seed` | Seed for the random train/val/test split

#### Example
In the movielens 1M dataset each line has the following format:
````
UserID::MovieID::Rating::Timestamp
````
To process it you have to specify the order of the columns, in this case uirt (for user, item, rating, timestamp), and the separator ("::"). If you want to use a hundred users for the validation set and a hundred others for the test set, you'll have to use the following command:
````
python preprocess.py -f path/to/datafile(.dat or .csv, etc) --columns uirt --sep :: --val_size 100 --test_size 100
````

