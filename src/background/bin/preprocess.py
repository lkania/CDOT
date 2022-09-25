import jax.numpy as np
import src.bin as bin_
from src.basis import bernstein as basis
from src.background.preprocess import int_omega as compute_int_omega
from src.dotdic import DotDic
from src.background.bin.delta import compute_sd


def preprocess(X, lower, upper, params):
    # assert params.k <= np.int32(params.bins / 2) * 2

    from_, to_ = bin_.bin(X, lower, upper, sections=np.int32(params.bins / 2))
    props = bin_.proportions(X, from_, to_)
    int_omega = compute_int_omega(params.k)
    M = basis.integrate(params.k, from_, to_)  # n_bins x n_parameters
    int_control = np.sum(M, axis=0).reshape(-1, 1)

    info = DotDic()
    info.name = params.name

    info.k = params.k

    info.bins = params.bins
    info.from_ = from_
    info.to_ = to_

    info.props = props
    info.int_omega = int_omega
    info.M = M
    info.int_control = int_control

    # parameters for delta ci
    info.n = X.shape[0]
    info.compute_sd = lambda data, jac: compute_sd(
        props=data,
        jac=jac,
        n=info.n)
    info.data_delta = props

    return info
