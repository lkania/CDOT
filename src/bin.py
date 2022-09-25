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
    counts_ = np.logical_and(X >= np.expand_dims(from_, 1),
                             X < np.expand_dims(to_, 1))
    counts_ = np.sum(counts_, 1)

    return counts_


@jit
def proportions(X, from_, to_):
    return counts(X, from_, to_) / len(X)


# TODO: allow for arbirary range and give number of bins as param
# check linspace command
def uniform_bin(step):
    core = np.arange(0, 1, step)
    from_ = core
    to_ = np.concatenate((core[1:], np.array([1])))

    return from_, to_
