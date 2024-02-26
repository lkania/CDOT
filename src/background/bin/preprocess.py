import jax.numpy as np
from src.bin import proportions
from src.background.bin.delta import influence
from functools import partial


def preprocess(params, method):
	empirical_probabilities, indicators = proportions(
		X=method.X,
		from_=params.from_,
		to_=params.to_)
	method.background.empirical_probabilities = empirical_probabilities

	int_omega = params.basis.int_omega(k=method.k)
	M = params.basis.integrate(method.k,
							   params.from_,
							   params.to_)  # n_bins x n_parameters
	int_control = np.sum(M, axis=0).reshape(-1, 1)

	method.background.int_omega = int_omega
	method.background.M = M
	method.background.int_control = int_control

	# indicators is a n_props x n_obs matrix that indicates
	# to which bin every observation belongs
	method.background.influence = partial(
		influence,
		empirical_probabilities=empirical_probabilities,
		indicators=indicators,
		grad=params.grad_op)
