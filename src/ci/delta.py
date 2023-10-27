from jax import numpy as np, jit
from jax.scipy.stats.norm import cdf, ppf as icdf


@jit
def delta_ci(point_estimate, influence, alpha=0.05):
    # influence must have shape = (n_parameters,n_obs)
    point_estimate = point_estimate.reshape(-1)
    n_parameters = point_estimate.shape[0]
    assert (n_parameters >= 1)

    mean_influence = np.mean(influence, axis=1).reshape(n_parameters, 1)

    centred_influence = influence - mean_influence  # n_parameters x n_obs
    # compute diagonal elements
    t2_hat = np.mean(np.square(centred_influence),
                     axis=1).reshape(-1)  # n_parameters

    n = influence.shape[1]
    std = np.sqrt(t2_hat)

    #######################################################
    # p-value
    #######################################################

    zscore = np.sqrt(n) * point_estimate / std
    pvalue = cdf(-zscore, loc=0, scale=1)

    #######################################################
    # lower confidence interval
    #######################################################

    delta = icdf(1 - alpha, loc=0, scale=1) * std / np.sqrt(n)
    ci = point_estimate - delta

    return ci, pvalue, zscore
