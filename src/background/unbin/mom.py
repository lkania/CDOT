import jax.numpy as np
from src.dotdic import DotDic
from src.basis import bernstein as basis
from src.background.preprocess import int_omega
from src.background.unbin.delta import compute_sd
from src.background.unbin.lambda_hat import compute_lambda_hat
from src.opt.jaxopt import normalized_nnls_with_linear_constraint


# from src.opt.error import ls_error


def fit(X, lower, upper, params):
    info = DotDic()
    info.k = params.k

    info.M = basis.outer_inner_product(k=params.k, a=lower, b=upper)
    info.int_omega = int_omega(params.k)
    info.int_control = basis.outer_integrate(k=params.k, a=lower, b=upper)
    info.n = X.shape[0]

    idx_control = np.array((X <= lower) + (X >= upper), dtype=np.bool_)
    X_control = X[idx_control]
    evals = basis.evaluate(k=params.k, X=X_control)
    info.mu = (np.sum(evals, axis=0) / info.n).reshape(-1)
    info.pc = np.sum(idx_control) / info.n
    info.mu2 = (np.sum(np.square(evals), axis=0) / info.n).reshape(-1)

    info.data_delta = (info.pc, info.mu)

    compute_gamma = lambda mu: normalized_nnls_with_linear_constraint(
        b=mu,
        A=info.M,
        c=info.int_omega,

        maxiter=params.maxiter,
        tol=params.tol,
        dtype=params.dtype)

    info.compute_lambda_hat = lambda data: compute_lambda_hat(
        pc=data[0],
        mu=data[1],

        compute_gamma=compute_gamma,
        int_control=info.int_control)

    info.compute_sd = lambda data, jac: compute_sd(
        pc=data[0],
        mu=data[1],

        jac_pc=jac[0],
        jac_mu=jac[1],

        n=info.n,
        mu2=info.mu2)

    # info.errors = lambda gamma: ls_error(b=info.mu, A=info.M, x=gamma)

    return info
