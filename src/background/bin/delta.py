import jax.numpy as np
from jax import jit


@jit
def compute_sd(props, jac, n):
    props = props.reshape(-1)

    D_hat = - np.outer(props, props)
    mask = 1 - np.eye(D_hat.shape[0])
    D_hat = D_hat * mask + np.diag(props * (1 - props))  # n_props x n_props

    sd = np.sqrt(np.sum((jac @ D_hat) * jac, axis=1) / n)

    return sd.reshape(-1)
