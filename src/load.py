import numpy as np
import jax.numpy as jnp


# load data function
def load(path):
    return jnp.array(np.loadtxt(path))
