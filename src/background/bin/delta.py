from jax import numpy as np


# Let func: R^{n_props} -> R^{p}
# influence returns R^{p} x n_observations


def influence(func, empirical_probabilities, indicators, grad):
	empirical_probabilities = empirical_probabilities.reshape(-1)
	n_probs = empirical_probabilities.shape[0]
	assert indicators.shape[0] == n_probs

	grad_op = grad(fun=func, argnums=0, has_aux=True)
	jac, aux = grad_op(empirical_probabilities)
	jac = jac.reshape(-1, n_probs)  # n_params x n_probs
	influence_ = jac @ indicators  # n_params x n_obs

	# NOTE: The following two lines are equivalent to
	#   influence_ = influence_ - np.mean(influence_,axis=1)
	# iff the empirical probabilities and indicators come from the
	# same empirical distribution
	# Conversely, this implementation allows for
	# the construction of de-biasing estimators when
	# indicators from another empirical distribution are provided
	centering = jac @ empirical_probabilities.reshape(n_probs, 1)
	centred_influence = influence_ - centering
	return centred_influence, aux


def t2_hat(func, empirical_probabilities, grad):
	props = empirical_probabilities.reshape(-1)
	n_probs = empirical_probabilities.shape[0]

	grad_op = grad(fun=func, argnums=0, has_aux=True)
	jac, aux = grad_op(empirical_probabilities)
	jac = jac.reshape(-1, n_probs)  # n_params x n_probs

	D_hat = - np.outer(props, props)
	mask = 1 - np.eye(D_hat.shape[0])
	D_hat = D_hat * mask + np.diag(props * (1 - props))  # n_props x n_props

	return np.sum((jac @ D_hat) * jac, axis=1), aux
