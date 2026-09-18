import jax.numpy as np
from jax import jit


def uniform_bin(lower, upper, n_bins):
	"""Create uniform control-region bins on [0,1] while omitting [lower, upper].
	
	Args:
	    lower: Float scalar in [0,1], lower signal-region boundary.
	    upper: Float scalar in [0,1], upper signal-region boundary.
	    n_bins: Integer scalar, nominal grid size used to form bin edges.
	Returns:
	    Tuple (from_, to_) of 1-D JAX arrays, each shape (B,), with control-bin endpoints."""
	assert lower >= 0 and upper <= 1
	l = np.linspace(start=0, stop=lower, num=int(n_bins * lower))
	u = np.linspace(start=upper, stop=1, num=int(n_bins * (1 - upper)))
	from_ = np.concatenate((l[:-1], u[:-1]))
	to_ = np.concatenate((l[1:], u[1:]))

	return from_, to_


# do not skip the signal region bin
def full_uniform_bin(n_bins, from_=0, to_=1):
	"""Create adjacent uniform bins over a complete interval without omitting a signal region.
	
	Args:
	    n_bins: Integer scalar, number of grid points (producing n_bins-1 intervals).
	    from_: Float scalar, lower domain endpoint.
	    to_: Float scalar, upper domain endpoint.
	Returns:
	    Tuple of 1-D JAX arrays, each shape (n_bins-1,), with lower and upper bin edges."""
	core = np.linspace(start=from_, stop=to_, num=n_bins)
	from_ = core[:-1]
	to_ = core[1:]

	return from_, to_


@jit
def indicator(X, from_, to_):
	"""Build observation-by-bin membership indicators for supplied intervals.
	
	Args:
	    X: JAX array, shape (N,), observations.
	    from_: JAX array, shape (B,), lower bin endpoints.
	    to_: JAX array, shape (B,), upper bin endpoints.
	Returns:
	    Integer JAX array, shape (B, N), with one for observations inside each half-open bin."""
	X = X.reshape(-1)
	# indicators has n_bins x n_obs
	indicators = np.logical_and(
		X >= np.expand_dims(from_, 1),
		X < np.expand_dims(to_, 1))
	indicators = np.array(indicators, dtype=np.int32)  # shape: n_probs x n_obs
	return indicators


@jit
def counts(X, from_, to_):
	"""Count observations in each supplied interval and return the membership matrix.
	
	Args:
	    X: JAX array, shape (N,), observations.
	    from_: JAX array, shape (B,), lower bin endpoints.
	    to_: JAX array, shape (B,), upper bin endpoints.
	Returns:
	    Tuple (counts, indicators) with shapes (B,) and (B,N), respectively."""
	X = X.reshape(-1)
	indicators = indicator(X, from_, to_)
	counts = np.sum(indicators, axis=1).reshape(-1)
	return counts, indicators