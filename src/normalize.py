import jax
import jax.numpy as np
from jax import jit
from functools import partial


@partial(jit, static_argnames=['tol'])
def safe_log(x, tol):
	"""Compute log(x) after flooring values at a numerical tolerance.
	
	Args:
	    x: Numeric/JAX array of any shape.
	    tol: Float scalar, positive floor applied before taking logarithms.
	Returns:
	    JAX array with the same shape as x."""
	x = np.where(x <= tol, tol, x)
	return np.log(x)


@partial(jit, static_argnames=['tol'])
def safe_ratio(num, den, tol):
	"""Compute num/den elementwise with stable handling near zero and near equality.
	
	Args:
	    num: Numeric/JAX scalar or array, broadcast-compatible with den.
	    den: Numeric/JAX scalar or array, broadcast-compatible with num.
	    tol: Float scalar, numerical threshold for zero/equality checks.
	Returns:
	    JAX array with the broadcast shape of num and den."""
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
	"""Normalize basis coefficients so their weighted integral over the domain equals one.
	
	Args:
	    gamma: JAX array, shape (P,), coefficients.
	    int_omega: JAX array, shape (P,), basis integrals over the full domain.
	Returns:
	    JAX array, shape (P,), normalized coefficients."""
	dot = np.dot(gamma.reshape(-1), int_omega.reshape(-1))
	return gamma / dot


@partial(jit, static_argnames=['tol'])
def threshold_non_neg(x, tol):
	"""Set entries not exceeding tol to zero while retaining larger nonnegative entries.
	
	Args:
	    x: Numeric/JAX array of any shape.
	    tol: Float scalar, cutoff threshold.
	Returns:
	    JAX array with the same shape and dtype as x."""
	return x * np.array(x > tol, dtype=x.dtype)


@partial(jit, static_argnames=['tol'])
def threshold(x, tol):
	"""Set entries with absolute magnitude at most tol to zero.
	
	Args:
	    x: Numeric/JAX array of any shape.
	    tol: Float scalar, absolute-value cutoff.
	Returns:
	    JAX array with the same shape and dtype as x."""
	return x * np.array(np.abs(x) > tol, dtype=x.dtype)