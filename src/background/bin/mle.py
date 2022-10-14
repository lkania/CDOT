from src.background.bin.preprocess import preprocess
# from jax.lax import fori_loop
import jax.numpy as np
from jax import jit
from src.normalize import normalize, threshold
# from src.opt.error import ls_error
from src.background.bin.lambda_hat import compute_lambda_hat
from jaxopt import FixedPointIteration, AndersonAcceleration
from functools import partial


# Three possibilities for update rules

@jit
def _delta1(gamma0, props, M, int_control, int_omega):
    return (M.transpose() @ (props.reshape(-1, 1) / (M @ gamma0.reshape(-1, 1)))) / int_control.reshape(-1, 1)


@jit
def _delta2(gamma0, props, M, int_control, int_omega):
    pred = np.dot(int_control.reshape(-1), gamma0.reshape(-1))
    denominator = int_control.reshape(-1, 1) + (np.sum(props) - pred) * int_omega.reshape(-1, 1)
    return (M.transpose() @ (props.reshape(-1, 1) / (M @ gamma0.reshape(-1, 1)))) / denominator


@jit
def _delta3(gamma0, props, M, int_control, int_omega):
    prob = np.dot(int_control.reshape(-1), gamma0.reshape(-1))
    denominator = (int_control.reshape(-1, 1) + (1 - prob) * int_omega.reshape(-1, 1))  # * np.sum(props)
    return (M.transpose() @ (props.reshape(-1, 1) / (M @ gamma0.reshape(-1, 1)))) / denominator


# @jit
# def _diff(gamma1, gamma2):
#     return np.max(np.abs(gamma1 - gamma2))


# @jit
# def _delta(props, M, gamma_and_diff, int_control):
#     gamma0, _ = gamma_and_diff
#     props = props.reshape(-1, 1)
#     gamma = gamma0 * ((M.transpose() @ (props / (M @ gamma0))) / int_control)
#     diff = _diff(gamma, gamma0)
#     return gamma, diff


# do 100 iterations of ci updates
# @jit
# def _iterate(props, M, gamma, int_control):
#     return fori_loop(
#         lower=0,
#         upper=500,
#         body_fun=lambda _, gamma_and_diff: _delta(
#             props=props,
#             M=M,
#             int_control=int_control,
#             gamma_and_diff=gamma_and_diff),
#         init_val=(gamma.reshape(-1, 1), 0))


# def _update_until_convergence(props, M, gamma0, int_control, tol=1e-4, maxiter=5):
#     it = 0
#     diff = 1
#     gamma = gamma0
#     while (diff > tol) and (it < maxiter):
#         gamma, diff = _iterate(props=props,
#                                M=M,
#                                gamma=gamma,
#                                int_control=int_control)
#         it += 1
#
#     return gamma, diff

@jit
def _update(gamma0, props, M, int_control, int_omega):
    return gamma0 * _delta2(gamma0=gamma0, props=props, M=M, int_control=int_control, int_omega=int_omega)


@partial(jit, static_argnames=['tol', 'maxiter', 'dtype'])
def _update_until_convergence(props, M, int_control, int_omega, tol, maxiter, dtype):
    sol = FixedPointIteration(fixed_point_fun=_update,
                              jit=True,
                              implicit_diff=True,
                              tol=tol,
                              maxiter=maxiter).run(
        np.full_like(int_control, 1, dtype=dtype) / np.sum(int_omega),  # init params (non-differentiable)
        props, M, int_control, int_omega)  # auxiliary parameters (differentiable)

    gamma = normalize(threshold(sol[0], tol=tol, dtype=dtype), int_omega=int_omega)
    return gamma, np.max(_delta2(gamma0=gamma, props=props, M=M, int_control=int_control, int_omega=int_omega) - 1)


def fit(X, lower, upper, params):
    info = preprocess(X=X, lower=lower, upper=upper, params=params)

    info.compute_gamma = lambda props: _update_until_convergence(props=props,
                                                                 M=info.M,
                                                                 int_control=info.int_control,
                                                                 int_omega=info.int_omega,
                                                                 tol=params.tol,
                                                                 maxiter=params.maxiter,
                                                                 dtype=params.dtype)

    info.compute_lambda_hat = lambda data: compute_lambda_hat(
        props=data,
        compute_gamma=info.compute_gamma,
        int_control=info.int_control)

    # info.errors = lambda gamma: ls_error(b=info.props_back, A=info.M, x=gamma)

    return info


# utility function for compute the negative log-likelihood of the original multinomial model
@jit
def nll(gamma, M, props, int_control):
    log_ratio = np.log((M @ gamma.reshape(-1, 1)).reshape(-1) / np.dot(gamma.reshape(-1), int_control.reshape(-1)))
    return -1 * np.sum(props.reshape(-1) * log_ratio)

# idea for directly optimizing the multinomial log-likelihood
# Note: current state: very unstable, poisson approximation is much better
# when further optimizing the results of the poisson approximation with the below optimizer
# there is no change, i.e. the poisson approixmation has found a stable local minima
# modify nll to be nll(gamma,data) data=(M, props, int_control)
# A = info.M
# b = info.props
# c = info.int_omega
# n_params = A.shape[1]
# pg = ProjectedGradient(fun=nll,
#                        verbose=False,
#                        acceleration=True,
#                        implicit_diff=True,
#                        tol=1e-6,
#                        maxiter=2000,
#                        jit=True,
#                        projection=lambda x, hyperparams: projection_polyhedron(x=x,
#                                                                                hyperparams=hyperparams,
#                                                                                check_feasible=False))
# # equality constraint
# A_ = c.reshape(1, -1)
# b_ = np.array([1.0])
#
# # inequality constraint
# G = -1 * np.eye(n_params)
# h = np.zeros((n_params,))
#
# pg_sol = pg.run(init_params=a[0].reshape(-1) / np.dot(a[0].reshape(-1), c.reshape(-1)),
#                 data=(A, b, info.int_control),
#                 hyperparams_proj=(A_, b_, G, h))
# x = pg_sol.params
