import numpy as np
import jax.numpy as jnp


def load_with_numpy(path):
	return np.loadtxt(path)


# load data function
def load(path):
	return jnp.array(load_with_numpy(path))
