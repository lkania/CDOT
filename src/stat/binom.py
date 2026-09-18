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
def __garwood_poisson_ci(n_events, alpha):
	"""Compute exact Garwood confidence limits for Poisson event counts.
	
	Args:
	    n_events: Numeric array of nonnegative counts, any shape.
	    alpha: Float scalar in (0,1), two-sided error probability.
	Returns:
	    Tuple (lower, upper) of arrays with the same shape as n_events."""
	assert np.sum(n_events < 0) == 0, "negative value found in n_events"

	L = np.where((n_events == 0),
				 0,
				 chi2.ppf(alpha / 2, 2 * n_events) / 2)
	U = np.where((n_events == 0),
				 chi2.ppf(1 - alpha, 2 * (n_events + 1)) / 2,
				 chi2.ppf(1 - alpha / 2, 2 * (n_events + 1)) / 2)

	return L, U


def __pooled_garwood_poisson_ci(n_events, alpha):
	# A confidence interval based on pooled data proceeds as follows
	# We will compute the CI using the sum of the observed events
	"""Compute a Garwood interval after pooling replicate Poisson counts along axis 0.
	
	Args:
	    n_events: Numeric array, shape (R, ...) with R replicate count vectors.
	    alpha: Float scalar in (0,1), error probability.
	Returns:
	    Tuple (lower, mean, upper), each a 1-D array over the pooled trailing entries."""
	n_observations = n_events.shape[0]
	sum_ = np.sum(n_events, axis=0)
	L, U = __garwood_poisson_ci(n_events=sum_, alpha=alpha)
	return (L.reshape(-1) / n_observations,
			sum_.reshape(-1) / n_observations,
			U.reshape(-1) / n_observations)


def garwood_poisson_ci(n_events, alpha, pool=False):
	"""Summarize Poisson counts with pooled or replicate-wise Garwood confidence intervals.
	
	Args:
	    n_events: Numeric array, shape (R, ...) for R replicates (or compatible count array).
	    alpha: Float scalar in (0,1), error probability.
	    pool: Boolean scalar; if True, pool counts over axis 0 before interval construction.
	Returns:
	    Tuple (lower, mean, upper) of 1-D arrays for each reported component."""
	if pool:
		# A confidence interval based on pooled data proceeds as follows
		# We compute the CI using the sum of the observed events
		Lq, mq, Uq = __pooled_garwood_poisson_ci(n_events=n_events, alpha=alpha)
	else:
		# If we are interested
		# not on pooling the data but on a confidence set of
		# confidence sets. Therefore, we will compute a Gaarwood CI for each
		# study and then proceed to aggreagate them

		L, U = __garwood_poisson_ci(n_events=n_events, alpha=alpha)
		Lq = np.quantile(L, q=alpha / 2, axis=0).reshape(-1)
		Uq = np.quantile(U, q=1 - alpha / 2, axis=0).reshape(-1)
		mq = np.mean(n_events, axis=0)

	assert not np.isnan(Lq).any()
	assert not np.isnan(Uq).any()
	assert not np.isnan(mq).any()
	assert np.all(mq <= Uq)
	assert np.all(mq >= Lq)

	return Lq, mq, Uq


def _scipy_cp(n_successes, n_trials, alpha, alternative='two-sided'):
	"""Compute Clopper-Pearson binomial intervals using SciPy's exact routine.
	
	Args:
	    n_successes: 1-D iterable/array, shape (Q,), success counts.
	    n_trials: Integer scalar, number of Bernoulli trials for each count.
	    alpha: Float scalar in (0,1), error probability.
	    alternative: String scalar specifying the interval alternative.
	Returns:
	    JAX array, shape (Q,2), lower and upper confidence limits."""
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
	"""Compute Clopper-Pearson binomial intervals with statsmodels.
	
	Args:
	    n_successes: Numeric array, shape (Q,), success counts.
	    n_trials: Integer scalar or shape-(Q,) array, corresponding trial counts.
	    alpha: Float scalar in (0,1), error probability.
	Returns:
	    Array, shape (Q,2), lower and upper confidence limits."""
	return np.stack(proportion_confint(
		count=n_successes,
		nobs=n_trials,
		alpha=alpha,
		method="beta"), axis=1)


def _clopper_pearson_binomial_ci(n_successes,
								 n_trials,
								 alpha):
	"""Validate inputs and compute exact Clopper-Pearson binomial intervals.
	
	Args:
	    n_successes: Numeric array, shape (Q,), nonnegative success counts.
	    n_trials: Integer scalar, number of trials, at least one.
	    alpha: Float scalar in (0,1), error probability.
	Returns:
	    Array, shape (Q,2), exact confidence limits."""
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
	"""Compute an exact confidence interval for the mean of binary simulation outcomes.
	
	Args:
	    values: 1-D array-like, shape (R,), binary 0/1 outcomes across simulations.
	    alpha: Float scalar in (0,1), error probability.
	Returns:
	    Tuple (lower, mean, upper) of scalar confidence summary values."""
	values_ = np.array(values, dtype=np.int32)
	cp = _clopper_pearson_binomial_ci(
		n_successes=[np.sum(values_)],
		n_trials=values_.shape[0],
		alpha=alpha)[0]
	mean = np.mean(values_)
	lower = cp[0]
	upper = cp[1]

	return lower, mean, upper


def normal_approximation_poisson_ratio_ci(X, Y, alpha, tol, pool=True):
	"""Construct normal-approximation intervals for ratios of paired Poisson counts.
	
	Args:
	    X: Numeric array, shape (R, Q) or compatible, numerator counts.
	    Y: Numeric array with the same shape as X, denominator counts.
	    alpha: Float scalar in (0,1), error probability.
	    tol: Float scalar, numerical tolerance for safe ratios.
	    pool: Boolean scalar; if True, pool replicates over axis 0.
	Returns:
	    Tuple (lower, mean, upper) of 1-D arrays, shape (Q,)."""
	assert X.shape == Y.shape

	if pool:
		lower, mean, upper = __pooled_normal_approximation_poisson_ratio_ci(
			X=X, Y=Y, alpha=alpha, tol=tol)
	else:
		lower, mean, upper = __normal_approximation_poisson_ratio_ci(
			X=X, Y=Y, alpha=alpha, tol=tol)
		lower = np.quantile(lower, q=alpha / 2, axis=0).reshape(-1)
		upper = np.quantile(upper, q=1 - alpha / 2, axis=0).reshape(-1)
		mean = np.mean(mean, axis=0)

	assert not np.isnan(lower).any()
	assert not np.isnan(upper).any()
	assert not np.isnan(mean).any()
	assert np.all(mean <= upper)
	assert np.all(mean >= lower)

	return lower, mean, upper


def __pooled_normal_approximation_poisson_ratio_ci(X, Y, alpha, tol):
	"""Pool replicate Poisson counts and compute a normal-approximation ratio interval.
	
	Args:
	    X: Numeric array, shape (R,Q), numerator counts.
	    Y: Numeric array, shape (R,Q), denominator counts.
	    alpha: Float scalar in (0,1), error probability.
	    tol: Float scalar, safe-ratio tolerance.
	Returns:
	    Tuple (lower, mean, upper), each shape (Q,)."""
	Xsum = np.sum(X, axis=0).reshape(-1)
	Ysum = np.sum(Y, axis=0).reshape(-1)
	return __normal_approximation_poisson_ratio_ci(X=Xsum,
												   Y=Ysum,
												   alpha=alpha, tol=tol)


def __normal_approximation_poisson_ratio_ci(X, Y, alpha, tol):
	"""Compute elementwise normal-approximation intervals for Poisson count ratios.
	
	Args:
	    X: Numeric scalar or array of any shape, numerator counts.
	    Y: Numeric scalar or array with the same shape as X, denominator counts.
	    alpha: Float scalar in (0,1), error probability.
	    tol: Float scalar, safe-ratio tolerance.
	Returns:
	    Tuple (lower, mean, upper) of arrays matching the broadcast shape of X and Y."""
	mean = normalize.safe_ratio(num=X, den=Y, tol=tol)
	# same as (Xbar/Ybar)^2 * (1 / Xbar + 1 / Ybar) * (1/n)
	# or equivalently Xsum / np.square(Ysum) + np.square(Xsum) / np.power(Ysum, 3)
	Y2 = np.square(Y)
	var = normalize.safe_ratio(num=X, den=Y2, tol=tol) * (1 + mean)
	sd = icdf(1 - alpha / 2, loc=0, scale=1) * np.sqrt(var)
	lower = normalize.threshold_non_neg(mean - sd, tol=0.0)
	upper = mean + sd
	return lower, mean, upper


# Note: we use the mean rather than refitting to all data
def boostrap_pivotal_ci(values, alpha):
	# values = np.array(values)
	"""Compute pivotal bootstrap confidence intervals from replicate estimates.
	
	Args:
	    values: Numeric array, shape (R, Q) or (R,), with R bootstrap replicates.
	    alpha: Float scalar in (0,1), error probability.
	Returns:
	    Tuple (lower, midpoint, upper) over the trailing estimate dimension(s)."""
	mean = np.mean(values, axis=0)
	lower = 2 * mean - np.quantile(values, q=1 - alpha / 2, axis=0)
	upper = 2 * mean - np.quantile(values, q=alpha / 2, axis=0)

	midpoint = (lower + upper) / 2

	return lower, midpoint, upper


def bootstrap_percentile_ci(values, alpha):
	"""Compute percentile bootstrap confidence intervals from replicate estimates.
	
	Args:
	    values: Numeric array, shape (R, Q) or (R,), with R bootstrap replicates.
	    alpha: Float scalar in (0,1), error probability.
	Returns:
	    Tuple (lower, midpoint, upper), each flattened to a 1-D array over estimates."""
	lower = np.quantile(values, q=alpha / 2, axis=0).reshape(-1)
	upper = np.quantile(values, q=1 - alpha / 2, axis=0).reshape(-1)
	midpoint = np.quantile(values, q=0.5, axis=0).reshape(-1)

	assert not np.isnan(lower).any()
	assert not np.isnan(upper).any()
	assert not np.isnan(midpoint).any()
	assert np.all(midpoint <= upper)
	assert np.all(midpoint >= lower)

	return lower, midpoint, upper