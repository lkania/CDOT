from src.background.bin.preprocess import preprocess
from src.opt.error import ls_error
from src.background.bin.lambda_hat import compute_lambda_hat
import jax.numpy as np


def fit(X, lower, upper, params):
    info = preprocess(X=X, lower=lower, upper=upper, params=params)

    compute_gamma = lambda props: params.nnls(b=props / np.sum(props.reshape(-1)), A=info.M, c=info.int_omega)

    info.compute_lambda_hat = lambda data: compute_lambda_hat(
        props=data,
        compute_gamma=compute_gamma,
        int_control=info.int_control)

    info.errors = lambda gamma: ls_error(b=info.props_back, A=info.M, x=gamma)

    return info
