import jax.numpy as np
from jax import jit


@jit
def _compute_lambda_hat(props, gamma, int_control):
    F_over_control = np.sum(props)
    B_over_control = np.dot(gamma.reshape(-1),
                            int_control.reshape(-1))
    return 1 - F_over_control / B_over_control


# M is a lxK matrix, integral of the basis over
# the the bins in the control region
def estimate_lambda(props, compute_gamma, int_control):
    gamma, gamma_aux = compute_gamma(props)
    lambda_hat = _compute_lambda_hat(props=props,
                                     gamma=gamma,
                                     int_control=int_control)
    return lambda_hat, (gamma, gamma_aux)
