import jax.numpy as np
from jax import jit
from jaxopt import FixedPointIteration, AndersonAcceleration
from functools import partial
from src.signal.delta import influence, objective


# TODO: The following method assumes that the signal
#  density is gaussian/normal. It can be generalized to
#  any signal by replacing the _delta iteration for a numerical optimization
@partial(jit, static_argnames=['signal'])
def _delta(data0, background_hat, X, lower, upper, signal):
    mu_hat0 = data0[0]
    sigma2_hat0 = data0[1]
    lambda_hat0 = data0[2]

    ##############################################################
    # E-step
    ##############################################################

    signal_density = signal(X=X, mu=mu_hat0, sigma2=sigma2_hat0)
    dens = lambda_hat0 * signal_density + (1 - lambda_hat0) * background_hat
    delta = lambda_hat0 * signal_density / dens

    ##############################################################
    # M-step
    ##############################################################

    lambda_hat = np.mean(delta)
    normalization = np.sum(delta)
    mu_hat = np.sum(delta * X) / normalization
    # restrict mean to signal region
    mu_hat = np.minimum(np.maximum(mu_hat, lower), upper)

    # TODO: restrict sigma2 with an upperbound
    sigma2_hat = np.sum(delta * np.square(X - mu_hat)) / normalization

    return np.array([mu_hat, sigma2_hat, lambda_hat])


# This function cannot be jitted when using jacrev
def _estimate_nu(lambda_hat0, background_hat, X, lower, upper, X_control,
                 signal, tol, maxiter):
    # compute initial parameters for EM
    mu_hat0 = np.mean(X_control)
    mu_hat0 = np.minimum(np.maximum(mu_hat0, lower), upper)

    # TODO: add restriction to sigma2_hat in the init parameter
    #  and during optimization
    sigma2_hat0 = np.mean(np.square(X_control - mu_hat0))

    # In order to avoid degenerate solutions,
    # restrict the initial lambda estimate
    lambda_hat0 = np.minimum(np.maximum(lambda_hat0, 0.01), 0.99)

    data0 = np.array([mu_hat0, sigma2_hat0, lambda_hat0])

    # compute fix point solution
    # TODO: We are ignoring the randomness of lambda_hat0, mu_hat0 and
    # sigma2_hat0 but since they
    # are initial parameters and the EM-optimization has a unique minimum,
    # it's unimportant
    __delta = lambda data0, background_hat: _delta(
        data0=data0,
        background_hat=background_hat,
        X=X.reshape(-1),
        lower=lower,
        upper=upper,
        signal=signal)

    sol = FixedPointIteration(
        fixed_point_fun=__delta,
        jit=True,
        implicit_diff=True,
        tol=tol,
        maxiter=maxiter).run(
        data0,  # init params are non-differentiable
        background_hat.reshape(-1))  # auxiliary parameters are differentiable

    nu_hat = sol[0]
    # where mu_hat = data[0],
    # sigma2_hat = data[1] and
    # lambda_hat = data[2]

    signal_error = (sol[1])[1]
    signal_fit = objective(X=X, nu=nu_hat, signal=signal,
                           background_hat=background_hat)
    signal_aux = (signal_error, signal_fit)

    return nu_hat, signal_aux


def estimate_nu(lambda_hat0, background_hat, params, method):
    # The following boolean array cannot be jitted
    idx_control = np.array(
        (method.X <= params.lower) + (method.X >= params.upper), dtype=np.bool_)
    X_control = method.X[idx_control].reshape(-1)
    return _estimate_nu(lambda_hat0=lambda_hat0,
                        background_hat=background_hat,
                        X=method.X,
                        lower=params.lower,
                        upper=params.upper,
                        X_control=X_control,
                        signal=params.signal.signal,
                        tol=params.tol,
                        maxiter=params.maxiter)


def fit(params, method):
    method.signal.estimate_nu = partial(estimate_nu, params=params,
                                        method=method)
    method.signal.influence = partial(influence, method=method, params=params)
