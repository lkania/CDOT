import jax.numpy as np
from jax.scipy.stats.beta import pdf as dbeta
from jax.scipy.special import betainc as pbeta

# needed for comb function
from jax.lax import add, sub, exp
from jax._src.scipy.special import gammaln

from jax import jit
from functools import partial


def comb(N, k):
	N_plus_1 = add(N, 1)
	k_plus_1 = add(k, 1)
	approx_comb = exp(
		sub(gammaln(N_plus_1),
			add(gammaln(k_plus_1),
				gammaln(sub(N_plus_1, k)))))
	return np.rint(approx_comb)


def bernstein_basis(N, k, x):
	return comb(N, k) * np.power(x, k) * np.power(1 - x, N - k)


# TODO: make into unit test
# check that evaluate correctly obtains the
# Bernstein basis
def check():
	xs = np.linspace(start=0, stop=1, num=100)
	for N in (np.arange(10) + 1):
		slow_comp = bernstein_basis(
			N,
			np.arange(N + 1),
			xs.reshape(-1, 1))
		fast_comp = evaluate(N, xs)
		assert np.max(np.abs(slow_comp - fast_comp)) < 1e-5


@partial(jit, static_argnames=['k'])
def evaluate(k, X):
	r = np.arange(0, k + 1).reshape(1, -1)
	den = dbeta(x=X.reshape(-1, 1), a=r + 1, b=k - r + 1)  # n x k
	den /= k + 1

	# Standardize so that int_Omega phi_k(x)=1
	# den /= k + 1

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
