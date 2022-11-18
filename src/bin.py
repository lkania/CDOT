import jax.numpy as np
from jax import jit


def split_positions(X, sections):
    X_sorted = np.sort(X)
    X_split = np.array_split(X_sorted, sections)

    # due to possible different lengths
    # it uses python base functions
    avgs = np.array(list(map(np.min, X_split[1:])))
    avgs += np.array(list(map(np.max, X_split[:-1])))
    avgs /= 2

    return avgs


def bin(X, lower, upper, sections):
    X_lower = X[X <= lower]
    X_upper = X[X >= upper]

    s_lower = split_positions(X_lower, sections)
    s_upper = split_positions(X_upper, sections)

    from_ = np.concatenate((np.array([0]), s_lower, np.array([upper]), s_upper))
    to_ = np.concatenate((s_lower, np.array([lower]), s_upper, np.array([1])))

    return from_, to_


@jit
def counts(X, from_, to_):
    X = X.reshape(-1)
    # indicators has n_bins x n_obs
    indicators = np.logical_and(X >= np.expand_dims(from_, 1), X < np.expand_dims(to_, 1))
    indicators = np.array(indicators, dtype=np.int32)  # shape: n_probs x n_obs
    counts = np.sum(indicators, axis=1).reshape(-1)
    return counts, indicators


@jit
def proportions(X, from_, to_):
    X = X.reshape(-1)
    n_obs = X.shape[0]
    counts_, indicators = counts(X, from_, to_)
    empirical_probabilities = counts_ / n_obs
    return empirical_probabilities, indicators


def uniform_bin(num, from_=0, to_=1):
    core = np.linspace(start=from_, stop=to_, num=num)
    from_ = core
    to_ = np.concatenate((core[1:], np.array([to_])))

    return from_, to_
