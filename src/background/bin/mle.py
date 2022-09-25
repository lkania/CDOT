from src.background.bin.preprocess import preprocess
from jax.lax import fori_loop
import jax.numpy as np
from jax import jit
from src.normalize import normalize
from src.opt.error import ls_error
from src.background.bin.lambda_hat import compute_lambda_hat


# TODO: replace by fixed point iteration

@jit
def _diff(gamma1, gamma2):
    return np.max(np.abs(gamma1 - gamma2))


@jit
def _delta(props, M, gamma_and_diff, int_control):
    gamma0, _ = gamma_and_diff
    props = props.reshape(-1, 1)
    gamma = gamma0 * ((M.transpose() @ (props / (M @ gamma0))) / int_control)
    diff = _diff(gamma, gamma0)
    return gamma, diff


# do 100 iterations of ci updates
@jit
def _iterate(props, M, gamma, int_control):
    return fori_loop(
        lower=0,
        upper=500,
        body_fun=lambda _, gamma_and_diff: _delta(
            props=props,
            M=M,
            int_control=int_control,
            gamma_and_diff=gamma_and_diff),
        init_val=(gamma.reshape(-1, 1), 0))


def _update_until_convergence(props, M, gamma0, int_control, tol=1e-4, maxiter=5):
    it = 0
    diff = 1
    gamma = gamma0
    while (diff > tol) and (it < maxiter):
        gamma, diff = _iterate(props=props,
                               M=M,
                               gamma=gamma,
                               int_control=int_control)
        it += 1

    return gamma, diff


def fit(X, lower, upper, params):
    info = preprocess(X=X, lower=lower, upper=upper, params=params)

    compute_gamma = lambda props: normalize(
        gamma_and_aux=_update_until_convergence(props=props,
                                                M=info.M,
                                                gamma0=np.full_like(info.int_control, 1),
                                                int_control=info.int_control),
        int_omega=info.int_omega)

    info.compute_lambda_hat = lambda data: compute_lambda_hat(
        props=data,
        compute_gamma=compute_gamma,
        int_control=info.int_control)

    info.errors = lambda gamma: ls_error(b=info.props_back, A=info.M, x=gamma)

    return info
