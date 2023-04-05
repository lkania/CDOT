from jax import jit
import jax.numpy as np


@jit
def ls_error(A, b, x):
    return (b.reshape(-1, 1) - A @ x.reshape(-1, 1)).reshape(-1)


@jit
def squared_ls_error(A, b, x):
    return np.sum(np.square(ls_error(A, b, x)))
