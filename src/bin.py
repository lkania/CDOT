import jax.numpy as np
from jax import jit


def split_positions(X, n_bins):
    X_sorted = np.sort(X)
    X_split = np.array_split(X_sorted, n_bins)

    # due to possible different lengths, it uses python base functions
    avgs = np.array(list(map(np.min, X_split[1:])))
    avgs += np.array(list(map(np.max, X_split[:-1])))
    avgs /= 2

    return avgs


# We compute the number of bins on the left sideband B_L
# and the right sideband B_R such that
#
# 1. L / B_L = R / B_R i.e. the number of observations per bin on the left and right is the same
# where L = number of observation on the left, R = number of observations on the right
#
# 2. B_L + B_R = N, where N is the total number of bins
#
# Solving the system gives
# B_R = R * N / (L+R) and B_L = L * N / (L+R)
def floor(a):
    return np.int32(np.floor(a))


def ceil(a):
    return np.int32(np.ceil(a))


def n_bins_(n_lower, n_upper, n_bins):
    n_control = n_lower + n_upper

    bins_lower = n_bins * n_lower / n_control
    bins_upper = n_bins * n_upper / n_control

    # we choose the ceil/floor combinations that minimizes the discrepancy
    # between L / B_L and R / B_R

    lower = np.array([floor(bins_lower), ceil(bins_lower)])
    upper = np.array([floor(bins_upper), ceil(bins_upper)])

    M = np.abs(np.subtract(lower.reshape(1, -1), upper.reshape(-1, 1)))
    index = np.unravel_index(np.argmin(M), shape=M.shape)  # 0 = floor, 1 = ceil

    func = lambda x: floor if x == 0 else ceil

    bins_lower = func(index[0])(bins_lower)
    bins_upper = func(index[1])(bins_upper)

    assert bins_lower > 0
    assert bins_upper > 0

    return bins_lower, bins_upper


# The method assumes that the data lays on the [0,1] interval
def adaptive_bin(X, lower, upper, n_bins):
    lower_idx = X <= lower
    n_lower = np.sum(lower_idx)
    X_lower = X[lower_idx]

    upper_idx = X >= upper
    n_upper = np.sum(upper_idx)
    X_upper = X[upper_idx]

    bins_lower, bins_upper = n_bins_(n_lower, n_upper, n_bins)
    s_lower = split_positions(X_lower, n_bins=bins_lower)
    s_upper = split_positions(X_upper, n_bins=bins_upper)

    from_ = np.concatenate((np.array([0]), s_lower, np.array([upper]), s_upper))
    to_ = np.concatenate((s_lower, np.array([lower]), s_upper, np.array([1])))

    return from_, to_


def uniform_bin(n_bins, from_=0, to_=1):
    core = np.linspace(start=from_, stop=to_, num=n_bins)
    from_ = core[:-1]
    to_ = core[1:]

    return from_, to_


@jit
def indicator(X, from_, to_):
    X = X.reshape(-1)
    # indicators has n_bins x n_obs
    indicators = np.logical_and(X >= np.expand_dims(from_, 1),
                                X < np.expand_dims(to_, 1))
    indicators = np.array(indicators, dtype=np.int32)  # shape: n_probs x n_obs
    return indicators


@jit
def counts(X, from_, to_):
    indicators = indicator(X, from_, to_)
    counts = np.sum(indicators, axis=1).reshape(-1)
    return counts, indicators


@jit
def proportions(X, from_, to_):
    X = X.reshape(-1)
    n_obs = X.shape[0]
    counts_, indicators = counts(X, from_, to_)
    empirical_probabilities = counts_ / n_obs
    return empirical_probabilities, indicators
