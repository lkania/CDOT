import jax.numpy as np
from jax.scipy.stats.beta import pdf as dbeta
from jax.scipy.special import betainc as pbeta

# needed for comb function
from jax.lax import add, sub, exp
from jax._src.scipy.special import gammaln

from jax import jit
from functools import partial


# approximate function
def comb(N, k):
    one = np.full(shape=k.shape, fill_value=1)
    N_plus_1 = add(N, one)
    k_plus_1 = add(k, one)
    approx_comb = exp(
        sub(gammaln(N_plus_1),
            add(gammaln(k_plus_1),
                gammaln(sub(N_plus_1, k)))))
    return np.rint(approx_comb)


@partial(jit, static_argnames=['k'])
def evaluate(k, X):
    r = np.arange(0, k + 1).reshape(1, -1)
    den = dbeta(x=X.reshape(-1, 1), a=r + 1, b=k - r + 1)  # n x k
    den /= k + 1

    # assert max(abs(np.sum(den,1)-1))<1e-5

    return den


# integrates the basis vector from a to b
@partial(jit, static_argnames=['k'])
def integrate(k, a, b):
    r = np.arange(0, k + 1).reshape(1, -1)
    lower = pbeta(x=a.reshape(-1, 1), a=r + 1, b=k - r + 1)
    upper = pbeta(x=b.reshape(-1, 1), a=r + 1, b=k - r + 1)
    int_ = upper - lower
    int_ /= k + 1

    return int_


def int_omega(k):
    return integrate(k, np.array([0]), np.array([1]))


@partial(jit, static_argnames=['k'])
def outer_integrate(k, a, b):
    return (1 / (k + 1)) - integrate(k=k, a=a, b=b)


# integrates inner product matrix from 0 to 1
@partial(jit, static_argnames=['k'])
def integrated_inner_product(k):
    r = np.arange(0, k + 1)
    sums = np.expand_dims(r, 1) + r
    int_ = pbeta(x=1, a=sums + 1, b=2 * k - sums + 1)

    # normalization
    c = comb(k, r)
    c = np.outer(c, c)
    c /= comb(2 * k, sums)
    c /= 2 * k + 1

    return int_ * c


# integrates inner product matrix from [0,a] and [b,1]
@partial(jit, static_argnames=['k'])
def outer_inner_product(k, a, b):
    r = np.arange(0, k + 1)
    sums = np.expand_dims(r, 1) + r
    lower = pbeta(x=a, a=sums + 1, b=2 * k - sums + 1)
    upper = pbeta(x=b, a=sums + 1, b=2 * k - sums + 1)
    int_ = 1 - (upper - lower)
    int_ = int_ * pbeta(x=1, a=sums + 1, b=2 * k - sums + 1)  # we cancel the normalization of pbeta

    # we introduce the normalization for the inner product
    c = comb(k, r)
    c = np.outer(c, c)
    c /= comb(2 * k, sums)
    c /= 2 * k + 1

    return int_ * c


def predict(gamma, k, from_, to_):
    gamma = gamma.reshape(-1, 1)
    basis_ = integrate(k=k, a=from_, b=to_)
    return basis_ @ gamma
