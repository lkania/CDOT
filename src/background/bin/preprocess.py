import jax.numpy as np
from src.bin import adaptive_bin, proportions, indicator, _adaptive_bin
from src.background.bin.delta import influence
from functools import partial
from src.background.density import background


def _influence(func,
			   empirical_probabilities,
			   indicators,
			   params,
			   from_,
			   to_,
			   X=None):
	idxs = indicators
	if X is not None:
		_, idxs = proportions(X=params.trans(X=X),
							  from_=from_,
							  to_=to_)
	return influence(func=func,
					 empirical_probabilities=empirical_probabilities,
					 indicators=idxs,
					 grad=params.grad_op)


def preprocess(params, method):
	# certify that there will be enough data-points to fit the background density
	assert method.k <= (params.bins + 1)

	method.tX = params.trans(X=method.X)

	# certify that all observations fall between 0 and 1
	assert (np.max(method.tX) <= 1) and (np.min(method.tX) >= 0)

	# TODO: We ignore the randomness of the equal-counts binning,
	#  it can be fixed by using a fixed binning
	from_, to_ = adaptive_bin(X=method.tX,
							  lower=params.tlower,
							  upper=params.tupper,
							  n_bins=params.bins)

	####################################################################
	# bookeeping
	####################################################################
	empirical_probabilities, indicators = proportions(X=method.tX,
													  from_=from_,
													  to_=to_)
	method.background.empirical_probabilities = empirical_probabilities

	int_omega = params.basis.int_omega(k=method.k)
	M = params.basis.integrate(method.k, from_, to_)  # n_bins x n_parameters
	int_control = np.sum(M, axis=0).reshape(-1, 1)

	method.background.bins = params.bins
	method.background.from_ = from_
	method.background.to_ = to_

	method.background.int_omega = int_omega
	method.background.M = M
	method.background.int_control = int_control

	# indicators is a n_props x n_obs matrix that indicates
	# to which bin every observation belongs
	method.background.influence = partial(
		_influence,
		params=params,
		from_=from_,
		to_=to_,
		empirical_probabilities=empirical_probabilities,
		indicators=indicators)

	method.background.estimate_background_from_gamma = partial(
		background,
		tilt_density=params.tilt_density,
		k=method.k,
		basis=params.basis)
