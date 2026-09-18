from jax import jit
from functools import partial
from src.basis import bernstein


@partial(jit, static_argnames=['k'])
def evaluate(k, X):
	"""Evaluate the normalized Bernstein basis whose elements each integrate to one.
	
	Args:
	    k: Integer scalar, polynomial order; P = k + 1 basis functions.
	    X: JAX array, shape (N,), points in [0,1].
	Returns:
	    JAX array, shape (N, P), normalized basis values."""
	return bernstein.evaluate(k, X) * (k + 1)


@partial(jit, static_argnames=['k'])
def integrate(k, a, b):
	"""Integrate each normalized Bernstein basis function over intervals [a,b].
	
	Args:
	    k: Integer scalar, polynomial order with P = k + 1.
	    a: JAX array, shape (B,), lower endpoints.
	    b: JAX array, shape (B,), matching upper endpoints.
	Returns:
	    JAX array, shape (B, P), normalized basis integrals."""
	return bernstein.integrate(k, a, b) * (k + 1)


@partial(jit, static_argnames=['k'])
def predict(gamma, k, from_, to_):
	"""Compute interval masses from normalized Bernstein coefficients.
	
	Args:
	    gamma: JAX array, shape (P,), normalized-basis coefficients.
	    k: Integer scalar, polynomial order, with P = k + 1.
	    from_: JAX array, shape (B,), lower interval endpoints.
	    to_: JAX array, shape (B,), upper interval endpoints.
	Returns:
	    JAX array, shape (B, 1), predicted interval masses."""
	return integrate(k=k, a=from_, b=to_) @ gamma.reshape(-1, 1)
