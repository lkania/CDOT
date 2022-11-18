import jax.numpy as np
from jax import jit


@jit
def _compute_lambda_hat(pc, gamma, int_control):
    return 1 - pc.reshape() / np.dot(gamma.reshape(-1), int_control.reshape(-1))


# compute_gamma might not be jittable
def compute_lambda_hat(pc, mu, compute_gamma, int_control):
    gamma, gamma_aux = compute_gamma(mu)
    lambda_hat = _compute_lambda_hat(pc=pc,
                                     gamma=gamma,
                                     int_control=int_control)
    return lambda_hat, gamma, gamma_aux
