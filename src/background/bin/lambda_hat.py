import jax.numpy as np
from jax import jit


@jit
def _compute_lambda_hat(props, gamma, int_control):
    return 1 - sum(props).reshape() / np.dot(gamma.reshape(-1),
                                             int_control.reshape(-1))


# M is a lxK matrix, integral of the basis over
# the the bins in the control region
# do not jit since not all compute_gamma functions are jittable
def compute_lambda_hat(props, compute_gamma, int_control):
    gamma, gamma_error = compute_gamma(props)
    lambda_hat = _compute_lambda_hat(props=props,
                                     gamma=gamma,
                                     int_control=int_control)
    return lambda_hat, gamma, gamma_error

#
# def compute_delta_ci(props, compute_gamma, int_control, n):
#     grad_op = value_and_grad(
#         lambda props: compute_lambda_hat(props=props,
#                                          compute_gamma=compute_gamma,
#                                          int_control=int_control),
#         argnums=[0], has_aux=True)  # grad w.r.t props
#
#     value_and_aux, grad = grad_op(props)
#     lambda_hat, gamma = value_and_aux
#     grad = grad[0]
#
#     delta_ci = _compute_delta_ci(props=props,
#                                  estimate=lambda_hat,
#                                  grad=grad,
#                                  n=n)
#     delta_ci = np.minimum(np.maximum(delta_ci, 0), 1)
#
#     return gamma, lambda_hat, delta_ci
