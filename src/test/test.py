from jax import numpy as np, random
import numpy as onp

######################################################################
# local libraries
######################################################################
from src.dotdic import DotDic
from src.test.builder import build
from src.ci.delta import delta_ci, _delta_ci


def test(args, X):
	return _test(params=build(args), X=np.array(X).reshape(-1))


def _test(params, X):
	method = DotDic()
	method.background = DotDic()
	method.model_selection = DotDic()
	method.k = params.k
	method.X = X

	# Certify that there will be enough data-points to fit the background density
	assert method.k <= (params.bins + 1)
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

	return method
