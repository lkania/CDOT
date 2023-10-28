#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np
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

        influence_, aux_ = method.background.influence(estimate)
        lambda_hat, aux = aux_
        gamma_hat, gamma_aux = aux

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
    # Test the null hypothesis
    #######################################################
    method.ci = ci
    # it returns one if the null hypothesis is rejected
    # at the alpha level, i.e. when ci > 0
    method.test = int(method.ci > params.tol)

    #######################################################
    # Store results
    #######################################################

    # pvalue
    method.pvalue = pvalue
    method.zscore = zscore

    # point estimates
    method.lambda_hat0 = lambda_hat
    method.gamma_hat = gamma_hat

    # background optimization statistics
    gamma_error, poisson_nll, multinomial_nll = gamma_aux
    method.gamma_error = gamma_error
    method.poisson_nll = poisson_nll
    method.multinomial_nll = multinomial_nll

    return method
