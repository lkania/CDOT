import os.path
from jax import numpy as np, random, jit
from functools import partial
from src.dotdic import DotDic
from src.load import load
import hasher


#######################################################

def load_weights(path, n_obs):
	if os.path.isfile(path):

		weights = load(path).reshape(-1)

		# check same dimension
		assert weights.shape[0] == n_obs

		# check that all events have strictly positive weight
		assert np.all(weights > 0)

		# force the weight to add up to one
		weights /= np.sum(weights)

	else:
		# if the weights are not provided, we uniform weights
		weights = np.ones(n_obs) / n_obs

	return weights


def load_background_and_signal(args):
	params = DotDic()

	#######################################################
	# Data parameters
	#######################################################
	params.data_id = args.data_id
	params.hash = hasher.hash_(params.data_id)
	params.cwd = args.cwd
	params.classifiers = args.classifiers

	#######################################################
	# load background data
	#######################################################
	params.path = '{0}/data/{1}'.format(params.cwd, params.data_id)
	print("Loading background data")
	params.background = DotDic()
	params.background.X = load(
		path='{0}/background/mass.txt'.format(params.path)).reshape(-1)

	params.background.weight = load_weights(
		path='{0}/background/weight.txt'.format(params.path),
		n_obs=params.background.X.shape[0])

	params.background.c = DotDic()
	for classifier in params.classifiers:
		params.background.c[classifier] = load(
			path='{0}/background/{1}.txt'.format(
				params.path, classifier)).reshape(-1)

		assert params.background.c[classifier].shape[0] == \
			   params.background.X.shape[0]

	print('Data source: {0} Size: {1}'.format(
		params.data_id, params.background.X.shape[0]))
	print("Data min={0} max={1}".format(
		np.round(np.min(params.background.X)),
		np.round(np.max(params.background.X))))

	#######################################################
	# Sampling
	#######################################################

	@partial(jit, static_argnames=['n', 'n_elements'])
	def choice(n, n_elements, probs, key):
		# if probs are assume to be uniform,
		# the following code is more efficient
		# return random.randint(key=key,
		# 						  minval=0,
		# 						  maxval=n_elements,
		# 						  shape=(n,)).reshape(-1)

		return random.choice(key=key,
							 a=n_elements,
							 shape=(n,),
							 p=probs,
							 replace=True)

	@partial(jit, static_argnames=['n', 'classifier'])
	def background_subsample(n, key, classifier):
		idx = choice(n=n,
					 n_elements=params.background.X.shape[0],
					 probs=params.background.weight,
					 key=key)
		X = params.background.X[idx].reshape(-1)
		c = params.background.c[classifier][idx].reshape(-1)

		return X, c

	print("Loading signal data")
	params.signal = DotDic()
	params.signal.X = load(
		path='{0}/signal/mass.txt'.format(params.path))

	params.signal.weight = load_weights(
		path='{0}/signal/weight.txt'.format(params.path),
		n_obs=params.signal.X.shape[0])

	params.signal.c = DotDic()
	for classifier in params.classifiers:
		params.signal.c[classifier] = load(
			path='{0}/signal/{1}.txt'.format(
				params.path,
				classifier)).reshape(-1)

		assert params.signal.c[classifier].shape[0] == params.signal.X.shape[0]

	#######################################################
	# Sampling
	#######################################################

	@partial(jit, static_argnames=['n', 'lambda_', 'classifier'])
	def subsample(n, lambda_, key, classifier):
		if lambda_ == 0:
			return background_subsample(
				n=n,
				classifier=classifier,
				key=key)

		# Note that the following function generates an array
		# whose size depends on n and lambda_
		# Therefore, we include n and lambda_ in the static_argnames
		key1, key2 = random.split(key, num=2)
		n_signal = int(n * lambda_)
		X, c = background_subsample(classifier=classifier,
									n=n - n_signal,
									key=key1)
		idx = choice(n=n_signal,
					 n_elements=params.signal.X.shape[0],
					 probs=params.signal.weight,
					 key=key2)
		signal_X = params.signal.X[idx].reshape(-1)
		signal_c = params.signal.c[classifier][idx].reshape(-1)

		X = np.concatenate((X, signal_X))
		c = np.concatenate((c, signal_c))

		return X, c

	def mask(X_, cutoff):
		X, c = X_
		return X, (c >= cutoff)

	def filter(X_, cutoff):
		X, c = X_
		return X[c >= cutoff]

	params.filter = filter

	@partial(jit, static_argnames=['n', 'classifier', 'lambda_'])
	def subsample_and_mask(n, classifier, lambda_, cutoff, key):
		X_ = subsample(n=n,
					   classifier=classifier,
					   lambda_=lambda_,
					   key=key)
		return mask(X_=X_, cutoff=cutoff)

	params.subsample_and_mask = subsample_and_mask

	return params
