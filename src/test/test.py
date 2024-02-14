from jax import numpy as np, random
import numpy as onp

######################################################################
# local libraries
######################################################################
from src.dotdic import DotDic
from src.test.builder import build
from src.ci.delta import delta_ci


def test(args, X):
	return _test(params=build(args), X=np.array(X).reshape(-1))


def _test(params, X):
	method = DotDic()
	method.background = DotDic()
	method.signal = DotDic()
	method.model_selection = DotDic()
	method.k = params.k

	method.X2 = None
	# if params.debias:
	#     idx = random.permutation(
	#         key=params.key,
	#         x=np.arange(X.shape[0]),
	#         independent=True)
	#     idxs = np.array_split(idx, indices_or_sections=2)
	#     X = X[idxs[0]]
	#     method.X2 = X[idxs[1]]

	method.X = X

	if method.k is None:
		method.model_selection.activated = True
		val_error = onp.zeros(len(params.ks))
		for i, k in enumerate(params.ks):
			method.k = k
			params.background.fit(params=params, method=method)
			val_error[i] = method.background.validation_error()
		k_star = onp.argmin(val_error)
		method.model_selection.val_error = val_error
		params.k = params.ks[k_star]
		method.k = params.k

	method.model_selection.activated = False
	params.background.fit(params=params, method=method)

	if not params.model_signal:
		#######################################################
		# Model free estimate
		#######################################################

		def estimate(h_hat):
			lambda_hat, aux = method.background.estimate_lambda(h_hat)
			return lambda_hat, (lambda_hat, aux)

		influence_, aux_ = method.background.influence(
			estimate, X=method.X2)
		lambda_hat, aux = aux_
		gamma_hat, gamma_aux = aux
		lambda_hat = lambda_hat + np.mean(influence_)

	else:
		#######################################################
		# Model based estimate
		#######################################################
		# The following function computes all the data necessary to compute
		# the delta-method confidence intervals for
		# lambda_hat0 (the model-free point estimate of lambda)
		# mu_hat (the model-based point estimate of mu)
		# sigma2_hat (the model-based point estimate of sigma2)
		# lambda_hat (the model-based point estimate of lambda)
		#######################################################
		params.signal.fit(params=params, method=method)
		influence_, aux = method.signal.influence()
		point_estimates, gamma_hat, gamma_aux, signal_aux = aux
		point_estimates = point_estimates.reshape(-1)

		#######################################################
		# In the following, we will only compute the CI for lambda_hat
		# However, if one wants to get the CIs for the other quantities,
		# Modify the function delta_ci so that it computes two-sided CIs
		# and execute
		#######################################################
		# delta_cis, stds = delta_ci(point_estimate=point_estimates,
		#                            influence=influence_)
		#######################################################
		# threshold confidence intervals
		#######################################################
		# lambda_hat0_ci = delta_cis[0, :]
		# mu_hat_delta = delta_cis[1, :]
		# sigma2_hat_ci = delta_cis[2, :]
		# lambda_hat_ci = delta_cis[3, :]
		#######################################################
		# point estimates
		#######################################################
		# lambda_hat0 = point_estimates[0]
		# mu_hat = point_estimates[1]
		# sigma2_hat = point_estimates[2]
		# lambda_hat = point_estimates[3]

		lambda_hat = point_estimates[3]
		influence_ = influence_[3, :].reshape(1, -1)

	#######################################################
	# compute one-sided confidence interval
	#######################################################
	ci, pvalue, zscore = delta_ci(
		point_estimate=np.array([lambda_hat]),
		influence=influence_)

	ci = ci[0]
	pvalue = pvalue[0]
	zscore = zscore[0]

	#######################################################
	# Store results
	#######################################################

	results = DotDic()
	results.k = method.k

	# pvalue
	results.ci = ci
	results.pvalue = pvalue
	results.zscore = zscore

	# point estimates
	results.lambda_hat = lambda_hat
	results.gamma_hat = gamma_hat

	# background optimization statistics
	gamma_error, poisson_nll, multinomial_nll = gamma_aux
	results.gamma_error = gamma_error
	results.poisson_nll = poisson_nll
	results.multinomial_nll = multinomial_nll

	return results
