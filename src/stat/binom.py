from jax import numpy as np
from scipy.stats._binomtest import _binom_exact_conf_int
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint


# See: https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
def exact_poisson_ci(n_events, alpha=0.05):
	n_events = n_events.reshape(-1)
	assert np.sum(n_events < 0) == 0, "negative value found in n_events"

	L = chi2.ppf(alpha / 2, 2 * n_events) / 2
	U = chi2.ppf(1 - alpha / 2, 2 * n_events + 2) / 2
	ci = np.stack([L, U], axis=1)
	return ci


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
	# Clopper-Pearson interval based on Beta distribution
	# See https://tedboy.github.io/statsmodels_doc/generated/statsmodels.stats.proportion.proportion_confint.html
	return np.stack(proportion_confint(
		count=n_successes,
		nobs=n_trials,
		alpha=alpha,
		method="beta"), axis=1)


def exact_binomial_ci(n_successes,
					  n_trials,
					  alpha=0.05):
	n_successes = np.array(n_successes)
	# check that n_successes and n_trials
	# contain no negative values
	assert np.sum(n_successes < 0) == 0, "negative value found in n_successes"
	assert n_trials >= 1, "n_trials must be greater or equal than 1"

	# compute confidence intervals
	return _statsmodels_cp(n_successes=n_successes,
						   n_trials=n_trials,
						   alpha=alpha)
