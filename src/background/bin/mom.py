from src.background.bin.preprocess import preprocess
# from src.opt.error import ls_error
from src.background.bin.lambda_hat import compute_lambda_hat
import jax.numpy as np
from src.opt.jaxopt import normalized_nnls_with_linear_constraint


def fit(X, lower, upper, params):
    info = preprocess(X=X, lower=lower, upper=upper, params=params)

    info.compute_gamma = lambda props: normalized_nnls_with_linear_constraint(b=props / np.sum(props),
                                                                              A=info.M,
                                                                              c=info.int_omega,

                                                                              maxiter=params.maxiter,
                                                                              tol=params.tol,
                                                                              dtype=params.dtype)

    info.compute_lambda_hat = lambda data: compute_lambda_hat(
        props=data,
        compute_gamma=info.compute_gamma,
        int_control=info.int_control)

    return info
