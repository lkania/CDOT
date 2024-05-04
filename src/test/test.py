from jax import numpy as np, jit
from functools import partial
from src.bin import proportions

######################################################################
# local libraries
######################################################################
from src.dotdic import DotDic
from src.test.builder import build as _build
from src.ci.delta import delta_ci, _delta_ci


def build(args):
	params = _build(args)

	return partial(_test, params=params)


@partial(jit, static_argnames=['params'])
def _test(params, X):
	# Certify that all observations fall between 0 and 1
	# since we are using Bernstein polynomials
	probs = X[:-1]
	n = X[-1]
	# X = np.array(X).reshape(-1)
	# assert (np.max(method.X) <= 1) and (np.min(method.X) >= 0)

	# empirical_probabilities, _ = proportions(
	# 	X=X,
	# 	from_=params.from_,
	# 	to_=params.to_)
	# assert not np.isnan(empirical_probabilities).any()

	#######################################################
	# Model free estimate
	#######################################################

	t2_hat, aux_ = params.background.t2_hat(
		func=params.background.estimate_lambda,
		empirical_probabilities=probs)

	lambda_hat, gamma_hat, gamma_aux = aux_

	# assert not np.isnan(gamma_hat).any()
	# assert not np.isnan(t2_hat)
	# assert not np.isnan(lambda_hat)

	#######################################################
	# compute one-sided confidence interval
	#######################################################
	ci, pvalue, zscore = _delta_ci(
		point_estimate=np.array([lambda_hat]),
		t2_hat=t2_hat,
		n=n)  # X.shape[0]

	# Slower method. Useful if further processing of Hadamard
	# derivatives is needed
	# if params.DEBUG:
	# 	influence_, _ = method.background.influence(estimate)
	# 	_, pvalue2, _ = delta_ci(
	# 		point_estimate=np.array([lambda_hat]),
	# 		influence=influence_)
	# 	assert pvalue == pvalue2

	ci = ci[0]
	pvalue = pvalue[0]
	zscore = zscore[0]

	# assert not np.isnan(ci)
	# assert not np.isnan(pvalue)
	# assert not np.isnan(zscore)

	#######################################################
	# Store results
	#######################################################
	method = dict()
	# method['params'] = params
	# data
	# method['X'] = X

	# # pvalue
	method['ci'] = ci
	method['pvalue'] = pvalue
	method['zscore'] = zscore
	#
	# # point estimates
	method['lambda_hat'] = lambda_hat
	method['gamma_hat'] = gamma_hat
	#
	# # background optimization statistics
	# gamma_error, poisson_nll, multinomial_nll = gamma_aux
	# method['gamma_error'] = gamma_error
	# method['poisson_nll'] = poisson_nll
	# method['multinomial_nll'] = multinomial_nll
	#
	# # set helper method
	# method.predict = partial(
	# 	params.basis.predict,
	# 	gamma=method.gamma_hat,
	# 	k=params.k)

	return method
