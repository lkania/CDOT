import jax.numpy as np
from src.background.bin.preprocess import preprocess
from src.normalize import normalize
from src.opt.error import ls_error
from src.background.bin.lambda_hat import compute_lambda_hat


def fit(X, lower, upper, params):
    # check that there are more bins than parameters

    info = preprocess(X=X, lower=lower, upper=upper, params=params)

    compute_gamma = lambda props: normalize(
        gamma_and_aux=params.nnls(b=np.sqrt(props),
                                  A=info.M / np.sqrt(props).reshape(-1, 1)),
        int_omega=info.int_omega)

    info.compute_lambda_hat = lambda data: compute_lambda_hat(
        props=data,
        compute_gamma=compute_gamma,
        int_control=info.int_control)

    info.errors = lambda gamma: ls_error(
        A=info.M / np.sqrt(info.props_back).reshape(-1, 1),
        b=np.sqrt(info.props_back),
        x=gamma)

    info.color = 'blue'

    return info
