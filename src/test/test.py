from jax import numpy as np, jit, grad
from functools import partial
# from src.bin import proportions
import jax
######################################################################
# local libraries
######################################################################
# from src.dotdic import DotDic
from src.test.builder import build as _build
from src.ci.delta import delta_ci, _delta_ci
from src.background.bin import mle
from jax.scipy.stats.norm import cdf, ppf as icdf


def build(args):
	params = _build(args)

	return partial(_test, params=params)


# Note that count array should be float to avoid issues
# when dividing two integers
@partial(jit, static_argnames=['params'])
def _test(params, counts, n):
	counts = counts.reshape(-1)
	props = counts / n

	#######################################################
	# Estimate without using a signal model
	#######################################################

	# _, aux = params.background.estimate_lambda(props)
	# lambda_hat = aux['lambda_hat']
	# gamma_hat = aux['gamma_hat']

	# TODO: wrong
	# efficient influence function
	# f = lambda props: params.background.loss(
	# 	gamma=gamma_hat, lambda_=lambda_hat, props=props)
	# grad_op = grad(fun=f, argnums=0, has_aux=False)
	# jac = grad_op(props)
	# jac = jac.reshape(-1)
	# influence = np.linalg.pinv(np.outer(jac, jac)) @ jac.reshape(-1, 1)
	# t2_hat_1 = np.sum(np.square(influence.reshape(-1)))
	# zscore_1 = np.sqrt(n) * lambda_hat / np.sqrt(t2_hat_1)

	# delta method for functional based on binning
	f = params.background.estimate_lambda
	grad_op = grad(fun=f, argnums=0, has_aux=True)
	jac, aux = grad_op(props)
	D_hat = - np.outer(props, props)
	mask = 1 - np.eye(D_hat.shape[0], dtype=props.dtype)
	D_hat = D_hat * mask + np.diag(props * (1 - props))
	t2_hat_2 = np.sum(
		(jac.reshape(1, -1) @ D_hat).reshape(-1) * jac.reshape(-1))

	lambda_hat = aux['lambda_hat']
	gamma_hat = aux['gamma_hat']
	zscore_2 = np.sqrt(n) * lambda_hat / np.sqrt(t2_hat_2)
	pvalue = 1 - cdf(zscore_2[0], loc=0, scale=1)

	# t2_hat2, _ = params.background.t2_hat(
	# 	func=params.background.estimate_lambda,
	# 	empirical_probabilities=probs)
	# t2_hat2 = t2_hat2[0]
	# zscore2 = np.sqrt(n) * lambda_hat / np.sqrt(t2_hat2)

	# delta method 2
	# t2_hat, aux = params.background.t2_hat(
	# 	func=partial(params.background.estimate_lambda2,
	# 				 init_lambda=lambda_hat,
	# 				 init_gamma=gamma_hat),
	# 	empirical_probabilities=probs)
	# lambda_hat = aux['lambda_hat']
	# gamma_hat = aux['gamma_hat']

	# jax.debug.print('t2 {t2_hat}', t2_hat=t2_hat)
	#######################################################
	# compute one-sided confidence interval
	#######################################################
	# ci, pvalue, zscore = _delta_ci(
	# 	point_estimate=np.array([lambda_hat]),
	# 	t2_hat=t2_hat,
	# 	n=n)

	# ci = ci[0]
	# pvalue = 0  # pvalue[0]

	#######################################################
	# Store results
	#######################################################
	method = dict()

	method['lambda_hat'] = lambda_hat
	method['gamma_hat'] = gamma_hat
	method['stat'] = pvalue

	# method['t2_hat1'] = t2_hat1
	# method['t2_hat_2'] = t2_hat_2
	# method['zscore1'] = zscore1
	# method['zscore2'] = zscore2

	return method
