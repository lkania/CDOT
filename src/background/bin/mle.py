from src.background.bin.preprocess import preprocess
import jax.numpy as np
from jax import jit
from src.normalize import normalize, threshold
from src.background.bin.lambda_hat import estimate_lambda
from jaxopt import FixedPointIteration
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


@jit
def _update(gamma0, props, M, int_control, int_omega):
    return gamma0 * _delta2(gamma0=gamma0, props=props, M=M, int_control=int_control, int_omega=int_omega)


@partial(jit, static_argnames=['dtype', 'tol', 'maxiter', 'dtype'])
def _update_until_convergence(props, M, int_control, int_omega, tol, maxiter, dtype):
    sol = FixedPointIteration(fixed_point_fun=_update,
                              jit=True,
                              implicit_diff=True,
                              tol=tol,
                              maxiter=maxiter).run(
        np.full_like(int_control, 1, dtype=dtype) / np.sum(int_omega),  # init params (non-differentiable)
        props, M, int_control, int_omega)  # auxiliary parameters (differentiable)

    gamma = normalize(threshold(sol[0], tol=tol, dtype=dtype), int_omega=int_omega)
    gamma_error = np.max(_delta2(gamma0=gamma, props=props, M=M, int_control=int_control, int_omega=int_omega) - 1)
    return gamma, gamma_error


def fit(params, method):
    preprocess(params=params, method=method)

    method.background.estimate_gamma = partial(_update_until_convergence,
                                               M=method.background.M,
                                               int_control=method.background.int_control,
                                               int_omega=method.background.int_omega,
                                               tol=params.tol,
                                               maxiter=params.maxiter,
                                               dtype=params.dtype)

    method.background.estimate_lambda = partial(estimate_lambda,
                                                compute_gamma=method.background.estimate_gamma,
                                                int_control=method.background.int_control)


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
