from jax import numpy as np, jit


@jit
def between(ci, lower, upper):
    return np.minimum(np.maximum(ci, lower), upper)


@jit
def between_0_and_1(ci):
    return between(ci, 0, 1)


@jit
def non_negative(ci):
    return np.maximum(ci, 0)


@jit
def delta_ci(point_estimate, influence):
    # influence must have shape = (n_parameters,n_obs)
    point_estimate = point_estimate.reshape(-1)
    n_parameters = point_estimate.shape[0]
    n = influence.shape[1]
    mean_influence = np.sum(influence, axis=1).reshape(n_parameters, 1) / n
    centred_influence = influence - mean_influence  # n_parameters x n_obs
    # compute diagonal elements
    t2_hat = np.sum(np.square(centred_influence), axis=1).reshape(-1) / n  # n_parameters
    delta = 1.96 * np.sqrt(t2_hat / n)
    return np.column_stack((np.subtract(point_estimate, delta), np.add(point_estimate, delta)))


def delta_cis(params, method):
    influence_, aux = method.signal.influence()
    point_estimates, gamma_hat, gamma_aux, signal_aux = aux
    point_estimates = point_estimates.reshape(-1)
    delta_cis_ = delta_ci(point_estimate=point_estimates, influence=influence_)

    # threshold confidence intervals
    method.lambda_hat0_delta = between_0_and_1(delta_cis_[0, :])
    method.mu_hat_delta = between(delta_cis_[1, :], lower=params.lower, upper=params.upper)
    method.sigma2_hat_delta = non_negative(delta_cis_[2, :])
    method.lambda_hat_delta = between_0_and_1(delta_cis_[3, :])

    # save results
    method.lambda_hat0 = point_estimates[0]
    method.mu_hat = point_estimates[1]
    method.sigma2_hat = point_estimates[2]
    method.lambda_hat = point_estimates[3]

    method.gamma = gamma_hat

    method.gamma_aux = gamma_aux
    method.signal_aux = signal_aux
