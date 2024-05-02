import jax.numpy as np
from jax import jit
from functools import partial


@jit
def normalize(gamma, int_omega):
	return gamma / np.dot(gamma.reshape(-1), int_omega.reshape(-1))


@partial(jit, static_argnames=['tol'])
def threshold_non_neg(x, tol):
	return x * np.array(x > tol, dtype=x.dtype)


@partial(jit, static_argnames=['tol'])
def threshold(x, tol):
	return x * np.array(np.abs(x) > tol, dtype=x.dtype)
