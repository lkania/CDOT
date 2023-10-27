#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np, jit
import numpy as onp

######################################################################
# local libraries
######################################################################
from src.dotdic import DotDic
from src.test.builder import build
from src.ci.delta import delta_ci


@jit
def _between(ci, lower, upper):
    return np.minimum(np.maximum(ci, lower), upper)


@jit
def _between_0_and_1(ci):
    return _between(ci, 0, 1)


@jit
def _non_negative(ci):
    return np.maximum(ci, 0)


def test(args, X):
    return _test(params=build(args), X=X)


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
        # Start of model agnostic estimate
        #######################################################

        def estimate(h_hat):
            lambda_hat, aux = method.background.estimate_lambda(h_hat)
            return lambda_hat, (lambda_hat, aux)

        influence_, aux_ = method.background.influence(estimate)
        lambda_hat, aux = aux_
        gamma_hat, gamma_aux = aux

        ci, pvalue, zscore = delta_ci(
            point_estimate=np.array([lambda_hat]),
            influence=influence_)
        ci = ci[0]
        pvalue = pvalue[0]
        zscore = zscore[0]
    else:
        # TODO: delta_ci only computes lower bounds now,
        # hence I need to update this section to only
        # compute ci for lambda
        # which makes sense, so that then one can replace the
        # signal by anything
        params.signal.fit(params=params, method=method)
        influence_, aux = method.signal.influence()
        point_estimates, gamma_hat, gamma_aux, signal_aux = aux
        point_estimates = point_estimates.reshape(-1)
        delta_cis, stds = delta_ci(point_estimate=point_estimates,
                                   influence=influence_)

        # threshold confidence intervals
        lambda_hat0_ci = delta_cis[0, :]
        mu_hat_delta = _between(
            delta_cis[1, :],
            lower=params.lower,
            upper=params.upper)
        sigma2_hat_ci = _non_negative(delta_cis[2, :])
        lambda_hat_ci = delta_cis[3, :]
        std = stds[3]

        # point estimates
        lambda_hat0 = point_estimates[0]
        mu_hat = point_estimates[1]
        sigma2_hat = point_estimates[2]
        lambda_hat = point_estimates[3]

    # pvalue
    method.pvalue = pvalue
    method.zscore = zscore
    
    # Test
    method.ci = ci
    # it returns one if the null hypothesis is rejected
    # at the alpha elvel, that is, when lower_ci > 0
    method.test = int(ci > params.tol)

    # point estimates
    method.lambda_hat0 = lambda_hat
    method.gamma_hat = gamma_hat

    # background optimization statistics
    gamma_error, poisson_nll, multinomial_nll = gamma_aux
    method.gamma_error = gamma_error
    method.poisson_nll = poisson_nll
    method.multinomial_nll = multinomial_nll

    return method
