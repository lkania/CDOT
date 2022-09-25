import jax.numpy as np
from jax import jit


@jit
def compute_sd(pc, mu, jac_pc, jac_mu, n, mu2):
    mu = mu.reshape(-1)
    mu2 = mu2.reshape(-1)
    pc_c = 1 - pc  # complement

    var_prop = np.square(jac_pc) * pc * pc_c

    D_mu_hat = - np.outer(mu, mu)
    mask = 1 - np.eye(D_mu_hat.shape[0])
    D_mu_hat = D_mu_hat * mask + np.diag(mu2 - np.square(mu))
    var_mu = np.sum((jac_mu @ D_mu_hat) * jac_mu, axis=1)

    D_mu_prop_hat = (mu * pc_c).reshape(-1, 1)
    var_mu_prop = (jac_mu @ D_mu_prop_hat).reshape(-1) * jac_pc

    var = var_prop + var_mu + 2 * var_mu_prop
    var /= n
    sd = np.sqrt(var)

    return sd

#
#
# def delta_ci(pc, mu, mu2, gamma, int_control, n):
#     grad_op = value_and_grad(
#         lambda pc, mu: lambda_hat(pc=pc,
#                                   mu=mu,
#                                   gamma=gamma,
#                                   int_control=int_control),
#         argnums=[0, 1])  # grad w.r.t pc and mu
#
#     lambda_hat_, grad_ = grad_op(pc, mu)
#
#     # concatenate grads
#     grad_pc = grad_[0]
#     grad_mu = grad_[1]
#
#     sd_ = sd(pc=pc, mu=mu, mu2=mu2, grad_pc=grad_pc, grad_mu=grad_mu, n=n)
#     delta = 1.96 * sd_
#
#     ci = np.array([lambda_hat_ - delta, lambda_hat_ + delta])
#     ci = ci.reshape(2, )
#
#     ci = np.maximum(ci, 0)
#     ci = np.minimum(ci, 1)
#
#     return lambda_hat_, ci
