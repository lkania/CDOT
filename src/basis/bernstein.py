import jax.numpy as np
from jax.scipy.stats.beta import pdf as dbeta
from jax.scipy.special import betainc as pbeta

from jax import jit
from functools import partial


@partial(jit, static_argnames=['k'])
def evaluate(k, X):
	r = np.arange(0, k + 1).reshape(1, -1)
	den = dbeta(x=X.reshape(-1, 1), a=r + 1, b=k - r + 1)  # n x k
	den /= k + 1

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


# def int_omega(k):
# 	# the following is equivalent to integrate(k, np.array([0]), np.array([1]))
# 	return np.full(shape=(1, k + 1), fill_value=1 / (k + 1))


def predict(gamma, k, from_, to_):
	return integrate(k=k, a=from_, b=to_) @ gamma.reshape(-1, 1)
