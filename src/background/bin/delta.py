import jax.numpy as np
from jax import jit, jacrev


# Let func: R^{n_props} -> R^{p}
# influence returns R^{p} x n_observations
def influence(func, empirical_probabilities, indicators):
    empirical_probabilities = empirical_probabilities.reshape(-1)
    n_probs = empirical_probabilities.shape[0]
    grad_op = jacrev(fun=func, argnums=0, has_aux=True)
    jac, aux = grad_op(empirical_probabilities)
    influence_ = jac.reshape(-1, n_probs) @ indicators.reshape(n_probs, -1)
    return influence_, aux

# @jit
# def compute_sd(props, jac, n):
#     props = props.reshape(-1)
#
#     D_hat = - np.outer(props, props)
#     mask = 1 - np.eye(D_hat.shape[0])
#     D_hat = D_hat * mask + np.diag(props * (1 - props))  # n_props x n_props
#
#     sd = np.sqrt(np.sum((jac @ D_hat) * jac, axis=1) / n)
#
#     return sd.reshape(-1)
