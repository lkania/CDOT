import jax
import jax.numpy as np
from jax import jit
from functools import partial


@partial(jit, static_argnames=['tol'])
def safe_ratio(num, den, tol):
	# In the following, we implement this decision tree
	# 	if num <= tol:
	# 		return 0.0
	# 	elif np.abs(num - den) <= tol:
	# 		return 1.0
	# 	elif den <= tol:
	# 		return num / tol
	# 	return num / den
	c1 = num <= tol
	c2 = np.abs(num - den) <= tol
	c3 = den <= tol
	ratio = np.where(c1, 0.0, num)
	ratio = np.where(np.logical_and(np.logical_not(c1), c2),
					 1.0,
					 ratio)
	c1_or_c2 = np.logical_or(c1, c2)
	ratio = np.where(np.logical_and(np.logical_not(c1_or_c2), c3),
					 num / tol,
					 ratio)
	c1_or_c2_or_c3 = np.logical_or(c1_or_c2, c3)
	safe_denominator = np.where(np.logical_not(c1_or_c2_or_c3),
								den,
								1.0)
	ratio = np.where(np.logical_not(c1_or_c2_or_c3),
					 num / safe_denominator,
					 ratio)
	return ratio


@jit
def normalize(gamma, int_omega):
	dot = np.dot(gamma.reshape(-1), int_omega.reshape(-1))
	return gamma / dot


@partial(jit, static_argnames=['tol'])
def threshold_non_neg(x, tol):
	return x * np.array(x > tol, dtype=x.dtype)


@partial(jit, static_argnames=['tol'])
def threshold(x, tol):
	return x * np.array(np.abs(x) > tol, dtype=x.dtype)
