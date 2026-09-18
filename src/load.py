import numpy as np
import jax.numpy as jnp


def load_with_numpy(path):
	"""Load a whitespace-delimited numeric text file as a NumPy array.
	
	Args:
	    path: String or path-like scalar identifying the input text file.
	Returns:
	    NumPy ndarray; dimension is determined by the file contents."""
	return np.loadtxt(path)


# load data function
def load(path):
	"""Load a numeric text file and convert it to a JAX array.
	
	Args:
	    path: String or path-like scalar identifying the input text file.
	Returns:
	    JAX array with the same dimensions as the numeric data in the file."""
	return jnp.array(load_with_numpy(path))
