import jax.numpy as np
from jax.scipy.stats.beta import pdf as dbeta
from jax.scipy.special import betainc as pbeta

from jax import jit
from functools import partial


@partial(jit, static_argnames=['k'])
def evaluate(k, X):
	"""Evaluate all Bernstein basis functions of order k at the supplied points.
	
	Args:
	    k: Integer scalar, polynomial order; the number of basis functions is P = k + 1.
	    X: JAX array, shape (N,), points in [0,1].
	Returns:
	    JAX array, shape (N, P), Bernstein basis values."""
	r = np.arange(0, k + 1).reshape(1, -1)
	den = dbeta(x=X.reshape(-1, 1), a=r + 1, b=k - r + 1)  # n x k
	den /= k + 1

	return den


# integrates the basis vector from a to b
@partial(jit, static_argnames=['k'])
def integrate(k, a, b):
	"""Integrate each Bernstein basis function over one or more intervals [a,b].
	
	Args:
	    k: Integer scalar, polynomial order with P = k + 1 basis functions.
	    a: JAX array, shape (B,), lower interval endpoints.
	    b: JAX array, shape (B,), matching upper endpoints.
	Returns:
	    JAX array, shape (B, P), basis integrals for each interval."""
	r = np.arange(0, k + 1).reshape(1, -1)
	lower = pbeta(x=a.reshape(-1, 1), a=r + 1, b=k - r + 1)
	upper = pbeta(x=b.reshape(-1, 1), a=r + 1, b=k - r + 1)
	int_ = upper - lower
	int_ /= k + 1

	return int_


def predict(gamma, k, from_, to_):
	"""Compute basis-expansion probability mass over specified intervals.
	
	Args:
	    gamma: JAX array, shape (P,), basis coefficients.
	    k: Integer scalar, polynomial order, with P = k + 1.
	    from_: JAX array, shape (B,), lower interval endpoints.
	    to_: JAX array, shape (B,), upper interval endpoints.
	Returns:
	    JAX array, shape (B, 1), predicted interval masses."""
	return integrate(k=k, a=from_, b=to_) @ gamma.reshape(-1, 1)