import jax.numpy as np
from src.signal.signal import fit_signal
from jax import jit, jacrev, jacfwd
from src.basis import bernstein as basis
from functools import partial


@partial(jit, static_argnames=['k'])
def density(gamma, k, x):
    den = basis.evaluate(k=k, X=x) @ gamma.reshape(-1, 1)
    return den.reshape(-1)


# NOTE: We are ignoring the randomness of lambda_hat0, mu_hat0 and sigma2_hat0 but since they
# are intial parameters and the EM-optimization has a unique minimum, it's unimportant
# TODO: We ignore the dependency of tilt_density on min(X), can be easily fixed by choosing a reasonable constant
# TODO: We ignore the randomness of the equal-counts binning, it can be fixed by using a fixed binning
# TODO: we ignore the randomness of the points at which the density is evaluated
@partial(jit, static_argnames=['k', 'compute_lambda_hat', 'tilt_density', 'tol', 'maxiter', 'dtype'])
def _estimate(data, k, X,
              lower, upper,
              mu_hat0, sigma2_hat0,
              compute_lambda_hat,
              tilt_density,
              tol,
              maxiter,
              dtype):
    X = X.reshape(-1)
    lambda_hat0, gamma, gamma_error = compute_lambda_hat(data=data)
    model_density = lambda x: density(gamma=gamma, x=x, k=k)
    background = tilt_density(model_density, X)

    mu_hat, sigma2_hat, lambda_hat, signal_error = fit_signal(
        X=X,
        mu_hat0=mu_hat0,
        sigma2_hat0=sigma2_hat0,
        lambda_hat0=lambda_hat0,
        background=background,
        lower=lower,
        upper=upper,
        tol=tol,
        maxiter=maxiter)

    estimates = np.array([lambda_hat0, lambda_hat, mu_hat, sigma2_hat],
                         dtype=dtype)

    return estimates, (lambda_hat0,
                       lambda_hat,
                       mu_hat,
                       sigma2_hat,
                       gamma,
                       gamma_error,
                       signal_error)


@jit
def between(ci, lower, upper):
    return np.minimum(np.maximum(ci, lower), upper)


@jit
def between_0_and_1(ci):
    return between(ci, 0, 1)


@jit
def non_negative(ci):
    return np.maximum(ci, 0)


def compute_delta_cis(estimates, jac, compute_sd, data_delta):
    estimates = estimates.reshape(-1)
    delta = 1.96 * compute_sd(data=data_delta, jac=jac)
    return np.column_stack((np.subtract(estimates, delta),
                            np.add(estimates, delta)))


def compute_cis(params, model):
    # compute initial parameters for EM
    idx_control = np.array(
        (params.X <= params.lower) + (params.X >= params.upper), dtype=np.bool_)
    X_control = params.X[idx_control].reshape(-1)
    mu_hat0 = np.mean(X_control)

    # restrict initial mean estimate to fall in the signal region
    mu_hat0 = np.minimum(np.maximum(mu_hat0, params.lower), params.upper)
    sigma2_hat0 = np.mean(np.square(X_control - mu_hat0))

    estimate = lambda data: _estimate(
        data=data,
        k=model.k,
        X=params.X,
        lower=params.lower,
        upper=params.upper,
        mu_hat0=mu_hat0,
        sigma2_hat0=sigma2_hat0,
        compute_lambda_hat=model.compute_lambda_hat,
        tilt_density=params.tilt_density,
        tol=params.tol,
        maxiter=params.maxiter,
        dtype=params.dtype)

    grad_op = jacrev(fun=estimate, argnums=0, has_aux=True)
    jac, aux = grad_op(model.data_delta)
    lambda_hat0, lambda_hat, mu_hat, sigma2_hat, gamma, gamma_error, signal_error = aux
    estimates = np.array([lambda_hat0, lambda_hat, mu_hat, sigma2_hat],
                         dtype=params.dtype)

    cis = compute_delta_cis(
        estimates=estimates,
        jac=jac,
        compute_sd=model.compute_sd,
        data_delta=model.data_delta)

    # threshold confidence intervals
    model.lambda_hat0_delta = between_0_and_1(cis[0, :])
    model.lambda_hat_delta = between_0_and_1(cis[1, :])
    model.mu_hat_delta = between(cis[2, :], lower=params.lower, upper=params.upper)
    model.sigma2_hat_delta = non_negative(cis[3, :])

    # save results
    model.lambda_hat0 = lambda_hat0
    model.lambda_hat = lambda_hat
    model.mu_hat = mu_hat
    model.sigma2_hat = sigma2_hat

    model.gamma = gamma

    model.gamma_error = gamma_error
    model.signal_error = signal_error
