from jax import numpy as np
from scipy.stats._binomtest import _binom_exact_conf_int
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint
from jax.scipy.stats.norm import ppf as icdf
from src import normalize


# See:
# Garwood, Frank (1936)
# https://arxiv.org/pdf/2104.05620
# - https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
# - https://stackoverflow.com/questions/14813530/poisson-confidence-interval-with-numpy
def _garwood_poisson_ci(n_events, alpha):
	# n_events = n_events.reshape(-1)
	assert np.sum(n_events < 0) == 0, "negative value found in n_events"

	L = np.where((n_events == 0),
				 0,
				 chi2.ppf(alpha / 2, 2 * n_events) / 2)
	U = np.where((n_events == 0),
				 chi2.ppf(1 - alpha, 2 * (n_events + 1)) / 2,
				 chi2.ppf(1 - alpha / 2, 2 * (n_events + 1)) / 2)

	return L, U


def garwood_poisson_ci(n_events, alpha):
	# if n_events.ndim == 1 or n_events.shape[0] == 1:
	# return _exact_poisson_ci(n_events=n_events, alpha=alpha)

	# A confidence interval based on pooled data proceeds as follows
	# We compute the CI using the sum of the observed events
	# n_observations = n_events.shape[0]
	# sum_ = np.sum(n_events, axis=0)
	# L, U = _garwood_poisson_ci(n_events=sum_.reshape(-1), alpha=alpha)
	# and obtain the CI for the average
	# return (L.reshape(-1) / n_observations,
	# 		sum_.reshape(-1) / n_observations,
	# 		U.reshape(-1) / n_observations)
	# However, we do not want this, since we are interested
	# not on pooling the data but on a confidence set of
	# confidence sets. Therefore, we will compute a Gaarwood CI for each
	# study and then proceed to aggreagate them

	L, U = _garwood_poisson_ci(n_events=n_events, alpha=alpha)
	Lq = np.quantile(L, q=alpha / 2, axis=0).reshape(-1)
	Uq = np.quantile(U, q=1 - alpha / 2, axis=0).reshape(-1)
	# midq = (Lq + Uq) / 2
	midq = np.mean(n_events, axis=0)

	assert not np.isnan(Lq).any()
	assert not np.isnan(Uq).any()
	assert not np.isnan(midq).any()
	assert np.all(midq <= Uq)
	assert np.all(midq >= Lq)

	return Lq, midq, Uq


def _scipy_cp(n_successes, n_trials, alpha, alternative='two-sided'):
	confidence_level = 1 - alpha
	return np.array(list(map(
		lambda k: _binom_exact_conf_int(
			k=k,  # number of successes
			n=n_trials,  # number of trials
			alternative=alternative,
			confidence_level=confidence_level),
		n_successes)))


def _statsmodels_cp(n_successes, n_trials, alpha):
	# Clopper-Pearson interval based on Beta distribution
	# See https://tedboy.github.io/statsmodels_doc/generated/statsmodels.stats.proportion.proportion_confint.html
	return np.stack(proportion_confint(
		count=n_successes,
		nobs=n_trials,
		alpha=alpha,
		method="beta"), axis=1)


def _clopper_pearson_binomial_ci(n_successes,
								 n_trials,
								 alpha):
	n_successes = np.array(n_successes)
	# check that n_successes and n_trials
	# contain no negative values
	assert np.any(n_successes >= 0), "negative value found in n_successes"
	assert n_trials >= 1, "n_trials must be greater or equal than 1"

	# compute confidence intervals
	return _statsmodels_cp(n_successes=n_successes,
						   n_trials=n_trials,
						   alpha=alpha)


# See: Clopper and Pearson (1934)
def clopper_pearson_binomial_ci(values, alpha):
	values_ = np.array(values, dtype=np.int32)
	cp = _clopper_pearson_binomial_ci(
		n_successes=[np.sum(values_)],
		n_trials=values_.shape[0],
		alpha=alpha)[0]
	mean = np.mean(values_)
	lower = cp[0]
	upper = cp[1]

	return lower, mean, upper


def normal_approximation_poisson_ratio_ci(X, Y, alpha, tol):
	assert X.shape[0] == Y.shape[0]
	Xsum = np.sum(X, axis=0).reshape(-1)
	Ysum = np.sum(Y, axis=0).reshape(-1)
	mean = normalize.safe_ratio(num=Xsum, den=Ysum, tol=tol)
	# same as (Xbar/Ybar)^2 * (1 / Xbar + 1 / Ybar) * (1/n)
	# or equivalently Xsum / np.square(Ysum) + np.square(Xsum) / np.power(Ysum, 3)
	Ysum2 = np.square(Ysum)
	var = normalize.safe_ratio(num=Xsum, den=Ysum2, tol=tol) * (1 + mean)
	sd = icdf(1 - alpha / 2, loc=0, scale=1) * np.sqrt(var)
	lower = normalize.threshold_non_neg(mean - sd, tol=0.0)
	upper = mean + sd
	return lower, mean, upper


# Note: we use the mean rather than refitting to all data
def boostrap_pivotal_ci(values, alpha):
	# values = np.array(values)
	mean = np.mean(values, axis=0)
	lower = 2 * mean - np.quantile(values, q=1 - alpha / 2, axis=0)
	upper = 2 * mean - np.quantile(values, q=alpha / 2, axis=0)

	midpoint = (lower + upper) / 2

	return lower, midpoint, upper


def bootstrap_percentile_ci(values, alpha, tol):
	values = np.array(values)

	lower = np.quantile(values, q=alpha / 2, axis=0)
	lower = normalize.threshold(lower, tol=tol)

	upper = np.quantile(values, q=1 - alpha / 2, axis=0)
	upper = normalize.threshold(upper, tol=tol)

	midpoint = np.quantile(values, q=0.5, axis=0)
	midpoint = normalize.threshold(midpoint, tol=tol)

	return lower, midpoint, upper
