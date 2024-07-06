from unittest import TestCase
from jax.lax import add, sub, exp
from jax._src.scipy.special import gammaln
import jax.numpy as np
import bernstein


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


class Test(TestCase):

	# Check that the fast computation of the Bernstein basis
	# is accurate enough for 10<=N<=90
	def test_evaluate(self):
		tol = 1e-5
		xs = np.linspace(start=0, stop=1, num=100)
		for N in range(10, 100, 10):
			slow_comp = bernstein_basis(
				N,
				np.arange(N + 1),
				xs.reshape(-1, 1))
			fast_comp = bernstein.evaluate(N, xs)
			self.assertTrue(np.max(np.abs(slow_comp - fast_comp)) < tol,
							"N={0}".format(N))
