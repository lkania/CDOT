import jax.numpy as np
from jax import jit
from functools import partial


@jit
def normalize(gamma, int_omega):
    return gamma / np.dot(gamma.reshape(-1), int_omega.reshape(-1))


@partial(jit, static_argnames=['tol', 'dtype'])
def threshold(gamma, tol, dtype):
    return gamma * np.array(gamma > tol, dtype=dtype)
