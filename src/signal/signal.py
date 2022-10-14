# from jax.lax import fori_loop
import jax.numpy as np
from jax import jit
from jax.scipy.stats.norm import pdf as dnorm
from jaxopt import FixedPointIteration
from functools import partial


# @jit
# def _diff(arg1, arg2):
#     return np.max(np.abs(arg1 - arg2))


@jit
def _delta(data0, background, X, lower, upper):
    mu_hat0 = data0[0]
    sigma2_hat0 = data0[1]
    lambda_hat0 = data0[2]

    ##############################################################
    # E-step
    ##############################################################

    signal = dnorm(X, loc=mu_hat0, scale=np.sqrt(sigma2_hat0)).reshape(-1)
    dens = lambda_hat0 * signal + (1 - lambda_hat0) * background
    delta = lambda_hat0 * signal / dens

    ##############################################################
    # M-step
    ##############################################################

    lambda_hat = np.mean(delta)
    normalization = np.sum(delta)
    mu_hat = np.sum(delta * X) / normalization
    # restrict mean to signal region
    mu_hat = np.minimum(np.maximum(mu_hat, lower), upper)

    sigma2_hat = np.sum(delta * np.square(X - mu_hat)) / normalization

    return np.array([mu_hat, sigma2_hat, lambda_hat])


# iterate ci updates
# @jit
# def _iterate(mu_hat, sigma2_hat, lambda_hat, X, background, lower, upper):
#     return fori_loop(
#         lower=0,
#         upper=1000,
#         body_fun=lambda _, data: _delta(data0=data,
#                                         background=background,
#                                         lower=lower,
#                                         upper=upper,
#                                         X=X),
#         init_val=np.array([mu_hat, sigma2_hat, lambda_hat, 0]))


# diff > tol means that the parameters changed by more
# than tol in the last iteration
# def _update_until_convergence(mu_hat, sigma2_hat, lambda_hat, X, background,
#                               lower, upper,
#                               tol=1e-4,
#                               maxiter=5):  # before maxiter 10
#     it = 0
#     diff = 1
#     while (diff > tol) and (it < maxiter):
#         data = _iterate(mu_hat=mu_hat,
#                         sigma2_hat=sigma2_hat,
#                         lambda_hat=lambda_hat,
#                         X=X,
#                         background=background,
#                         lower=lower,
#                         upper=upper)
#         mu_hat = data[0]
#         sigma2_hat = data[1]
#         lambda_hat = data[2]
#         diff = data[3]
#         it += 1
#
#     return mu_hat, sigma2_hat, lambda_hat, diff

@partial(jit, static_argnames=['tol', 'maxiter'])
def fit_signal(X, mu_hat0, sigma2_hat0, lambda_hat0, background, lower, upper, tol, maxiter):
    # in order to avoid degenerate solutions, restrict the initial lambda estimate
    lambda_hat0 = np.minimum(np.maximum(lambda_hat0, 0.01), 0.99)

    # restrict initial mean estimate to fall in the signal region
    mu_hat0 = np.minimum(np.maximum(mu_hat0, lower), upper)

    data0 = np.array([mu_hat0, sigma2_hat0, lambda_hat0])

    # compute fix point solution
    sol = FixedPointIteration(fixed_point_fun=_delta,
                              jit=True,
                              implicit_diff=True,
                              tol=tol,
                              maxiter=maxiter).run(
        data0,  # init params (non-differentiable)
        background.reshape(-1), X.reshape(-1), lower, upper)  # auxiliary parameters (differentiable)

    data = sol[0]
    mu_hat = data[0]
    sigma2_hat = data[1]
    lambda_hat = data[2]
    diff = (sol[1])[1]

    return mu_hat, sigma2_hat, lambda_hat, diff
