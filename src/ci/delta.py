from jax import numpy as np, jit
from jax.scipy.stats.norm import cdf, ppf as icdf


@jit
def delta_ci(point_estimate, influence, alpha=0.05):
    # influence must have shape = (n_parameters,n_obs)
    point_estimate = point_estimate.reshape(-1)
    n_parameters = point_estimate.shape[0]
    assert n_parameters >= 1
    assert influence.shape[0] == n_parameters

    # compute diagonal elements
    t2_hat = np.mean(np.square(influence), axis=1).reshape(-1)  # n_parameters

    n = influence.shape[1]
    std = np.sqrt(t2_hat)

    #######################################################
    # p-value corresponding to the one-sided test
    #######################################################

    zscore = np.sqrt(n) * point_estimate / std
    pvalue = cdf(-zscore, loc=0, scale=1)

    #######################################################
    # one-sided lower confidence interval
    #######################################################

    delta = icdf(1 - alpha, loc=0, scale=1) * std / np.sqrt(n)
    ci = point_estimate - delta

    return ci, pvalue, zscore
