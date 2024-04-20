from jax import numpy as np, random
#######################################################
# Utilities
#######################################################
from src.dotdic import DotDic
from src.load import load


#######################################################

def load_background(args):
	params = DotDic()
	params.seed = int(args.seed)
	params.key = random.PRNGKey(seed=params.seed)

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

	def new_key():
		key, subkey = random.split(params.key)
		params.key = subkey
		return params.key

	params.new_key = new_key

	def choice(n, n_elements):
		return random.choice(
			params.new_key(),
			np.arange(n_elements),
			shape=(n,)).reshape(-1)

	params.choice = choice

	def _subsample(n, classifier=None):
		idx = params.choice(n=n,
							n_elements=params.background.X.shape[0])
		X = params.background.X[idx].reshape(-1)
		if classifier is None:
			return X

		c = params.background.c[classifier][idx].reshape(-1)

		return X, c

	params.background.subsample = _subsample

	def subsample(n, lambda_, classifier=None):
		if lambda_ > 0:
			raise ValueError('lambda > 0 but no signal has been loaded')
		return params.background.subsample(n=n, classifier=classifier)

	params.subsample = subsample

	def filter(X_, cutoff):
		X, c = X_
		X = X[c >= cutoff]
		X = random.permutation(
			key=params.new_key(),
			x=X,
			independent=False)
		return X

	params.filter = filter

	def subsample_and_filter(n, classifier, lambda_, cutoff):
		return params.filter(X_=params.subsample(n=n,
												 classifier=classifier,
												 lambda_=lambda_),
							 cutoff=cutoff)

	params.subsample_and_filter = subsample_and_filter

	return params


def load_signal(params):
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

	def subsample(n, lambda_, classifier=None):
		if lambda_ == 0:
			return params.background.subsample(n=n,
											   classifier=classifier)

		assert classifier is not None

		n_signal = np.int32(n * lambda_)
		X, c = params.background.subsample(classifier=classifier,
										   n=n - n_signal)
		idx = params.choice(n=n_signal,
							n_elements=params.signal.X.shape[0])
		signal_X = params.signal.X[idx].reshape(-1)
		signal_c = params.signal.c[classifier][idx].reshape(-1)

		X = np.concatenate((X, signal_X))
		c = np.concatenate((c, signal_c))

		return X, c

	params.subsample = subsample

	return params
