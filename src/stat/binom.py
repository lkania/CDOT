import numpy as np

from scipy.stats._binomtest import _binom_exact_conf_int as _clopper_pearson, _binom_wilson_conf_int as _wilson


def vectorize(obs, f):
    n_estimators = obs.shape[1]
    cis = np.zeros((n_estimators, 2))
    for i in np.arange(n_estimators):
        cis[i, :] = f(obs[:, i])
    return cis


def clopper_pearson(obs,
                    alternative='two-sided',
                    alpha=0.05):
    confidence_level = 1 - alpha
    return vectorize(obs, f=lambda o: _clopper_pearson(
        k=np.sum(o),  # number of successes
        n=o.shape[0],  # number of trials
        alternative=alternative,
        confidence_level=confidence_level))


def wilson(obs,
           alternative='two-sided',
           confidence_level=0.95,
           correction=False):
    return vectorize(obs, f=lambda o: _wilson(
        k=np.sum(o),  # number of successes
        n=o.shape[0],  # number of trials
        alternative=alternative,
        confidence_level=confidence_level,
        correction=correction))
