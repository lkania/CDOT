from jax import numpy as np, random, jit
from functools import partial
from src.dotdic import DotDic
from src.load import load


#######################################################

def load_background_and_signal(args):
	params = DotDic()
	params.hash = args.hash

	#######################################################
	# Data parameters
	#######################################################
	params.cwd = args.cwd
	params.data_id = args.data_id
	params.classifiers = args.classifiers

	#######################################################
	# load background data
	#######################################################
	params.path = '{0}/data/{1}'.format(params.cwd, params.data_id)
	print("Loading background data")
	params.background = DotDic()
	params.background.X = load(
		path='{0}/background/mass.txt'.format(params.path))
	params.background.c = DotDic()
	for classifier in params.classifiers:
		params.background.c[classifier] = load(
			path='{0}/background/{1}.txt'.format(
				params.path, classifier))

	print('Data source: {0} Size: {1}'.format(
		params.data_id, params.background.X.shape[0]))
	print("Data min={0} max={1}".format(
		np.round(np.min(params.background.X)),
		np.round(np.max(params.background.X))))

	#######################################################
	# Sampling
	#######################################################
	@partial(jit, static_argnames=['n', 'n_elements'])
	def choice(n, n_elements, key):
		return random.randint(key=key,
							  minval=0,
							  maxval=n_elements,
							  shape=(n,)).reshape(-1)

	# params.choice = choice

	@partial(jit, static_argnames=['n', 'classifier'])
	def background_subsample(n, key, classifier):
		idx = choice(n=n,
					 n_elements=params.background.X.shape[0],
					 key=key)
		X = params.background.X[idx].reshape(-1)
		c = params.background.c[classifier][idx].reshape(-1)

		return X, c

	# params.background.subsample = _subsample

	# def subsample(n, lambda_, key, classifier=None):
	# 	# if lambda_ > 0:
	# 	# 	raise ValueError('lambda > 0 but no signal has been loaded')
	# 	return params.background.subsample(n=n,
	# 									   classifier=classifier,
	# 									   key=key)
	#
	# params.subsample = subsample

	# def filter(X_, cutoff):
	# 	X, c = X_
	# 	X = X[c >= cutoff]
	# 	# X = random.permutation(
	# 	# 	key=params.new_key(),
	# 	# 	x=X,
	# 	# 	independent=False)
	# 	return X
	#
	# def subsample_and_filter(n, classifier, lambda_, cutoff, key):
	# 	return filter(X_=params.subsample(n=n,
	# 									  classifier=classifier,
	# 									  lambda_=lambda_,
	# 									  key=key),
	# 				  cutoff=cutoff)
	#
	# params.subsample_and_filter = subsample_and_filter
	print("Loading signal data")
	params.signal = DotDic()
	params.signal.X = load(
		path='{0}/signal/mass.txt'.format(params.path))
	params.signal.c = DotDic()
	for classifier in params.classifiers:
		params.signal.c[classifier] = load(
			path='{0}/signal/{1}.txt'.format(
				params.path,
				classifier))

	#######################################################
	# Sampling
	#######################################################

	# Note that this function will generate an array
	# whose size depends on n and lambda_
	@partial(jit, static_argnames=['n', 'lambda_', 'classifier'])
	def subsample(n, lambda_, key, classifier):
		if lambda_ == 0:
			return background_subsample(
				n=n,
				classifier=classifier,
				key=key)

		# assert classifier is not None

		key1, key2 = random.split(key, num=2)
		n_signal = int(n * lambda_)
		X, c = background_subsample(classifier=classifier,
									n=n - n_signal,
									key=key1)
		idx = choice(n=n_signal,
					 n_elements=params.signal.X.shape[0],
					 key=key2)
		signal_X = params.signal.X[idx].reshape(-1)
		signal_c = params.signal.c[classifier][idx].reshape(-1)

		X = np.concatenate((X, signal_X))
		c = np.concatenate((c, signal_c))

		return X, c

	# params.subsample = subsample

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
