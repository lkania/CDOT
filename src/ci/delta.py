from jax import numpy as np, jit


# TODO: replace 1.96 by z-score from alpha
@jit
def delta_ci(point_estimate, influence):
    # influence must have shape = (n_parameters,n_obs)
    point_estimate = point_estimate.reshape(-1)
    n_parameters = point_estimate.shape[0]
    assert (n_parameters >= 1)
    n = influence.shape[1]
    mean_influence = np.sum(influence, axis=1).reshape(n_parameters, 1) / n

    centred_influence = influence - mean_influence  # n_parameters x n_obs
    # compute diagonal elements
    t2_hat = np.sum(np.square(centred_influence), axis=1).reshape(
        -1) / n  # n_parameters

    std = np.sqrt(t2_hat / n)
    delta = 1.96 * std
    cis = np.column_stack(
        (np.subtract(point_estimate, delta),
         np.add(point_estimate, delta)))
    return cis, std
