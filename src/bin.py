import jax.numpy as np
from jax import jit


def uniform_bin(lower, upper, n_bins):
	assert lower >= 0 and upper <= 1
	l = np.linspace(start=0, stop=lower, num=int(n_bins * lower))
	u = np.linspace(start=upper, stop=1, num=int(n_bins * (1 - upper)))
	from_ = np.concatenate((l[:-1], u[:-1]))
	to_ = np.concatenate((l[1:], u[1:]))

	return from_, to_


# do not skip the signal region bin
def full_uniform_bin(n_bins, from_=0, to_=1):
	core = np.linspace(start=from_, stop=to_, num=n_bins)
	from_ = core[:-1]
	to_ = core[1:]

	return from_, to_


@jit
def indicator(X, from_, to_):
	X = X.reshape(-1)
	# indicators has n_bins x n_obs
	indicators = np.logical_and(
		X >= np.expand_dims(from_, 1),
		X < np.expand_dims(to_, 1))
	indicators = np.array(indicators, dtype=np.int32)  # shape: n_probs x n_obs
	return indicators


@jit
def counts(X, from_, to_):
	X = X.reshape(-1)
	indicators = indicator(X, from_, to_)
	counts = np.sum(indicators, axis=1).reshape(-1)
	return counts, indicators

# @jit
# def proportions(X, from_, to_, n=None):
# 	if n is None:
# 		n = X.shape[0]
# 	X = X.reshape(-1)
# 	counts_, indicators = counts(X, from_, to_)
# 	empirical_probabilities = counts_ / n
# 	return empirical_probabilities, indicators
