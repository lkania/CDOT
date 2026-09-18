from jax import numpy as np, jit
from functools import partial


@partial(jit, static_argnames=['rate', 'base'])
def _trans(X, rate, base):
	"""Evaluate the exponential factor used by the mass transformation.
	
	Args:
	    X: Numeric/JAX array of any shape, input values.
	    rate: Float scalar, positive exponential rate.
	    base: Float scalar, lower-location shift.
	Returns:
	    JAX array with the same shape as X."""
	return np.exp(-rate * (X - base))


@partial(jit, static_argnames=['rate', 'base'])
def trans(X, rate, base, scale):
	"""Map values from the original mass scale to the normalized transformed scale.
	
	Args:
	    X: Numeric/JAX array of any shape, original-scale values.
	    rate: Float scalar, exponential rate.
	    base: Float scalar, lower-location shift.
	    scale: Float scalar, normalization constant for the target interval.
	Returns:
	    JAX array with the same shape as X."""
	return (1 - _trans(X=X, rate=rate, base=base)) / scale


def safe_trans(X, rate, base, scale):
	"""Apply the exponential transformation after checking that all inputs exceed the base.
	
	Args:
	    X: Numeric/JAX array of any shape, values satisfying X >= base.
	    rate: Float scalar, exponential rate.
	    base: Float scalar, minimum allowed input value.
	    scale: Float scalar, normalization constant.
	Returns:
	    JAX array with the same shape as X on the transformed scale."""
	assert np.all(X >= base), ('X values should greater or equal to base. '
							   'X contains value {0} but base={1}'.format(
		np.min(X), base))
	return trans(X=X, rate=rate, base=base, scale=scale)


@partial(jit, static_argnames=['rate', 'base'])
def inv_trans(X, rate, base, scale):
	"""Invert the normalized exponential mass transformation.
	
	Args:
	    X: Numeric/JAX array of any shape, transformed values.
	    rate: Float scalar, exponential rate.
	    base: Float scalar, location shift used in the forward transform.
	    scale: Float scalar, normalization constant used in the forward transform.
	Returns:
	    JAX array with the same shape as X on the original scale."""
	return np.log(1 - X * scale) / (-rate) + base


# we are computing density(trans(X=X,c=c,base=base)) * c * _trans(X=X,c=c,base=base)
# avoiding re-computation
@partial(jit, static_argnames=['density', 'rate', 'base'])
def tilt_density(density, X, rate, base, scale):
	"""Apply the Jacobian correction to a density under the exponential transformation.
	
	Args:
	    density: Callable mapping a shape-(N,) transformed array to shape-(N,) density values.
	    X: JAX array, shape (N,), original-scale evaluation points.
	    rate: Float scalar, exponential rate.
	    base: Float scalar, location shift.
	    scale: Float scalar, transformation normalization constant.
	Returns:
	    JAX array, shape (N,), density values on the original scale."""
	trans_aux = _trans(X=X, rate=rate, base=base)
	trans_ = (1 - trans_aux) / scale
	return density(X=trans_) * rate * trans_aux / scale


def transform(rate, a, b):
	"""Construct forward, density-tilt, and inverse exponential transformation callables.
	
	Args:
	    rate: Float scalar, exponential rate.
	    a: Float scalar, lower endpoint/base of the original domain.
	    b: Float scalar or None, upper endpoint used to scale to [0,1]; None leaves scale equal to one.
	Returns:
	    Tuple of three callables (forward, tilted-density, inverse), each acting elementwise on arrays."""
	base = a
	scale = 1 if b is None else (1 - _trans(X=b, rate=rate, base=base))
	return partial(trans, rate=rate, base=base, scale=scale), \
		partial(tilt_density, rate=rate, base=base, scale=scale), \
		partial(inv_trans, rate=rate, base=base, scale=scale)
