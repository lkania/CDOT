from jax import numpy as np
from scipy.stats._binomtest import _binom_exact_conf_int
from statsmodels.stats.proportion import proportion_confint


def _scipy_cp(n_successes, n_trials, alpha=0.05, alternative='two-sided'):
    confidence_level = 1 - alpha
    return np.array(list(map(
        lambda k: _binom_exact_conf_int(
            k=k,  # number of successes
            n=n_trials,  # number of trials
            alternative=alternative,
            confidence_level=confidence_level),
        n_successes)))


def _statsmodels_cp(n_successes, n_trials, alpha=0.05):
    return np.stack(proportion_confint(
        count=n_successes,
        nobs=n_trials,
        alpha=alpha,
        method="beta"), axis=1)


def clopper_pearson(n_successes,
                    n_trials,
                    alpha=0.05):
    n_successes = np.array(n_successes)
    # check that n_successes and n_trials
    # contain no negative values
    idx = np.where(n_successes < 0)[0]
    assert len(idx) == 0, "negative value [{0}] found in n_successes".format(
        n_successes[idx[0]])
    assert n_trials >= 1, "n_trials must be greater or equal than 1"

    # compute confidence intervals
    return _statsmodels_cp(n_successes=n_successes,
                           n_trials=n_trials,
                           alpha=alpha)
