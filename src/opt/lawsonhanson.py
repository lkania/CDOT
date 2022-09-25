import jax.numpy as np
from jax import jit
from src.opt.error import squared_ls_error
from src.opt.jax import ls


def _alpha(d, s):
    return np.min(d / (d - s))


# compute A.transpose() (b - A @ d)
@jit
def _w(Ab, AA, d):
    return Ab - AA @ d


# the algorithm just looks for the best subset
# A is L x M
# b is L x 1
def nnls(A, b, tol=1e-11):
    # in order to match scipy implementation
    maxiter = np.maximum(A.shape[0], A.shape[1]) * 4

    # pre-computations
    b = b.reshape(-1, 1)
    Ab = A.transpose() @ b
    AA = A.transpose() @ A

    # init
    L, M = A.shape
    P = np.zeros(M, dtype=np.bool_)  # R = not P = ~P
    d = np.zeros((M, 1), dtype=np.float64)  # TODO: change this via params
    w = A.transpose() @ b

    #  utils
    range_ = np.arange(0, M)
    zero_vec = np.zeros((M, 1))

    it = 0
    while (not np.all(P)) and (
            np.max(w[~P, :].reshape(-1)) > tol) and it < maxiter:

        m_ = np.argmax(w.reshape(-1)[~P])  # max error from i not in P
        m = range_[~P][m_]

        P = P.at[m].set(True)

        s_P = ls(A[:, P], b).reshape(-1, 1)
        s = zero_vec.at[P, :].set(s_P)

        while np.any(s_P < -tol):
            # if i is not in P, the s_i = 0
            # we select i : i in P and s_i < - tol
            idx = s.reshape(-1) < -tol
            alpha = _alpha(d=d[idx, :], s=s[idx, :])

            d = d + alpha * (s - d)

            # move out of P, all indices in P such that d_i = 0
            P = P.at[(np.abs(d).reshape(-1) < tol) * P].set(False)

            s_P = ls(A[:, P], b).reshape(-1, 1)
            s = zero_vec.at[P, :].set(s_P)
            it += 1

        d = s
        w = _w(Ab=Ab, AA=AA, d=d)

        it += 1

    return d, squared_ls_error(A=A, b=b, x=d)
