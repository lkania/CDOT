from src.background.bin.preprocess import preprocess
from src.normalize import normalize, threshold
from src.background.bin.lambda_hat import estimate_lambda

from jax import jit, numpy as np
from jaxopt import FixedPointIteration, AndersonAcceleration, ProjectedGradient
from functools import partial
from jaxopt.projection import projection_polyhedron


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


@partial(jit, static_argnames=['_delta'])
def _update(gamma0, props, M, int_control, int_omega, _delta):
    return gamma0 * _delta(gamma0=gamma0, props=props, M=M, int_control=int_control, int_omega=int_omega)


@partial(jit, static_argnames=['dtype', 'tol', 'maxiter', 'dtype'])
def _update_until_convergence1(props, M, int_control, int_omega, tol, maxiter, dtype):
    _delta = _delta2
    sol = AndersonAcceleration(fixed_point_fun=partial(_update, _delta=_delta),
                               jit=True,
                               implicit_diff=True,
                               tol=tol,
                               maxiter=maxiter).run(
        np.full_like(int_control, 1, dtype=dtype) / np.sum(int_omega),  # init params (non-differentiable)
        props, M, int_control, int_omega)  # auxiliary parameters (differentiable)

    gamma = normalize(threshold(sol[0], tol=tol, dtype=dtype), int_omega=int_omega)

    gamma_error = np.max(
        np.abs(_delta(gamma0=gamma, props=props, M=M, int_control=int_control, int_omega=int_omega) - 1))
    gamma_fit = multinomial_nll(gamma=gamma, data=(M, props, int_control))
    gamma_aux = (gamma_error, gamma_fit)

    return gamma, gamma_aux


@partial(jit, static_argnames=['dtype', 'tol', 'maxiter', 'dtype'])
def _update_until_convergence2(props, M, int_control, int_omega, tol, maxiter, dtype):
    A = M
    b = props.reshape(-1)
    c = int_omega.reshape(-1)
    n_params = A.shape[1]
    pg = ProjectedGradient(fun=poisson_nll,
                           verbose=False,
                           acceleration=True,
                           implicit_diff=True,
                           tol=tol,
                           maxiter=maxiter,
                           jit=True,
                           projection=lambda x, hyperparams: projection_polyhedron(x=x,
                                                                                   hyperparams=hyperparams,
                                                                                   check_feasible=False))
    # equality constraint
    A_ = c.reshape(1, -1)
    b_ = np.array([1.0])

    # inequality constraint
    G = -1 * np.eye(n_params)
    h = np.zeros((n_params,))

    pg_sol = pg.run(init_params=np.full_like(c, 1, dtype=dtype) / n_params,
                    data=(A, b, c),
                    hyperparams_proj=(A_, b_, G, h))
    x = pg_sol.params

    gamma = normalize(threshold(x, tol=tol, dtype=dtype), int_omega=int_omega)

    gamma_error = 0
    gamma_fit = multinomial_nll(gamma=gamma, data=(M, props, int_control))
    gamma_aux = (gamma_error, gamma_fit)

    return gamma, gamma_aux


def fit(params, method):
    preprocess(params=params, method=method)

    method.background.estimate_gamma = partial(_update_until_convergence1,
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
def multinomial_nll(gamma, data):
    M, props, int_control = data
    log_ratio = np.log((M @ gamma.reshape(-1, 1)).reshape(-1) / np.dot(gamma.reshape(-1), int_control.reshape(-1)))
    return (-1) * np.sum(props.reshape(-1) * log_ratio)


@jit
def poisson_nll(gamma, data):
    M, props, int_control = data
    log_preds = np.log((M @ gamma.reshape(-1, 1)).reshape(-1))
    int_over_control = np.dot(gamma.reshape(-1), int_control.reshape(-1))
    return (-1) * (np.sum(props.reshape(-1) * log_preds) - int_over_control)

# idea for directly optimizing the multinomial log-likelihood
# Note: current state: very unstable, poisson approximation is much better
# when further optimizing the results of the poisson approximation with the below optimizer
# there is no change, i.e. the poisson approixmation has found a stable local minima
# modify nll to be nll(gamma,data) data=(M, props, int_control)
