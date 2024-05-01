from jax import numpy as np, random
import numpy as onp
from functools import partial

######################################################################
# local libraries
######################################################################
from src.dotdic import DotDic
from src.test.builder import build as _build
from src.ci.delta import delta_ci, _delta_ci


def build(args):
	params = _build(args)
	method = DotDic()
	method.background = DotDic()
	method.model_selection = DotDic()
	method.k = params.k

	params.background.preprocess(params=params, method=method)

	return partial(_test, params=params, method=method)


def _test(params, method, X):
	method.X = np.array(X).reshape(-1)

	# Certify that all observations fall between 0 and 1
	# since we are using Bernstein polynomials
	assert (np.max(method.X) <= 1) and (np.min(method.X) >= 0)

	params.background.fit(params=params, method=method)

	#######################################################
	# Model free estimate
	#######################################################

	def estimate(h_hat):
		lambda_hat, aux = method.background.estimate_lambda(h_hat)
		return lambda_hat, (lambda_hat, aux)

	t2_hat, aux_ = method.background.t2_hat(estimate)
	lambda_hat, aux = aux_
	gamma_hat, gamma_aux = aux

	assert not np.isnan(gamma_hat).any()
	assert not np.isnan(t2_hat)
	assert not np.isnan(lambda_hat)

	#######################################################
	# compute one-sided confidence interval
	#######################################################
	ci, pvalue, zscore = _delta_ci(
		point_estimate=np.array([lambda_hat]),
		t2_hat=t2_hat,
		n=method.X.shape[0])

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

	assert not np.isnan(ci)
	assert not np.isnan(pvalue)
	assert not np.isnan(zscore)

	#######################################################
	# Store results
	#######################################################
	method.params = params

	# pvalue
	method.ci = ci
	method.pvalue = pvalue
	method.zscore = zscore

	# point estimates
	method.lambda_hat = lambda_hat
	method.gamma_hat = gamma_hat

	# background optimization statistics
	gamma_error, poisson_nll, multinomial_nll = gamma_aux
	method.gamma_error = gamma_error
	method.poisson_nll = poisson_nll
	method.multinomial_nll = multinomial_nll

	# set helper method
	method.background.predict = partial(
		params.basis.predict,
		gamma=method.gamma_hat,
		k=method.k)

	return method
