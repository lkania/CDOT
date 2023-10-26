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

    def estimate(h_hat):
        lambda_hat0, aux = method.background.estimate_lambda(h_hat)
        return lambda_hat0, (lambda_hat0, aux)

    influence_, aux_ = method.background.influence(estimate)
    lambda_hat0, aux = aux_
    gamma_hat, gamma_aux = aux
    gamma_error, poisson_nll, multinomial_nll = gamma_aux

    # compute confidence interval

    ci, aux = delta_ci(
        point_estimate=np.array([lambda_hat0]),
        influence=influence_)
    std = aux[0]

    # Test
    method.delta_ci = ci[0]
    method.test = not ((method.delta_ci[0] <= 0) and (0 <= method.delta_ci[1]))

    # save results
    method.lambda_hat0 = lambda_hat0
    method.std = std
    method.gamma_hat = gamma_hat
    method.gamma_error = gamma_error
    method.poisson_nll = poisson_nll
    method.multinomial_nll = multinomial_nll

    return method
