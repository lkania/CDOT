import jax.numpy as np
from jax.lax import stop_gradient
from scipy.optimize import nnls as snnls
from src.opt.error import squared_ls_error
from src.opt.jax import ls


def _nnls(A, b, active):
    M = A.shape[1]
    x = np.zeros(M).at[active].set(ls(A=A[:, active], b=b).reshape(-1))
    return x.reshape(-1, 1), squared_ls_error(A=A, b=b, x=x)


def _active_variables(A, b, tol=1e-11):
    maxiter = np.maximum(A.shape[0], A.shape[1]) * 4
    return np.array(snnls(A=A, b=b.reshape(-1), maxiter=maxiter)[0] > tol)


# NOTE: the gradient from this function isn't useful since
# it doesn't take into account the effect of the data on the
# variable selection step
def nnls(A, b, tol=1e-11):
    active = _active_variables(A=stop_gradient(A), b=stop_gradient(b), tol=tol)
    return _nnls(A=A, b=b, active=active)
