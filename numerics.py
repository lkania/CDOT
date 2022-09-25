#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as np
import jax.random as random
from jax import jit
from jaxopt import FixedPointIteration
#######################################################

from src.load import load
from src.dotdic import DotDic
from experiments.evaluate import run

# nnls optimizers
from src.opt.scipy import nnls as lawson_scipy_nnls

# from src.opt.lawsonhanson import nnls as lawson_jax_nnls
# from src.opt.jaxopt import nnls as pg_jaxopt_nnls
# from src.opt.pgd import nnls as pg_jax_nnls
# from src.opt.cvx import nnls as conic_cvx_nnls

params = DotDic()
params.seed = 0
params.key = random.PRNGKey(seed=params.seed)
params.k = 30  # high impact on jacobian computation for non-bin methods
params.bins = 44  # high impact on jacobian computation for bin methods
params.data_id = 1
params.data = './data/{0}/m_muamu.txt'.format(params.data_id)
params.folds = 200
params.dtype = np.float64
params.name = 'test'

# numerical methods
params.tol = 1e-5
params.maxiter = 10000

# %%

#######################################################
# load background data
#######################################################
X = load(params.data)

# %%
# least squares numerics

# transform data
from src.transform import transform

trans, tilt_density = transform(X)
tX = trans(X)
lower = trans(450 - 3 * 20)
upper = trans(450 + 3 * 20)
# %%

from src.background.bin.preprocess import preprocess

info = preprocess(X=tX, lower=lower, upper=upper, params=params)

# %%

from src.normalize import normalize
from src.opt.jaxopt import nnls as pg_jaxopt_nnls
from src.opt.jaxopt import nnls_with_linear_constraint

x1 = normalize(gamma_and_aux=pg_jaxopt_nnls(b=info.props, A=info.M), int_omega=info.int_omega)

x2 = nnls_with_linear_constraint(b=info.props, A=info.M, c=info.int_omega)

x3 = nnls_with_linear_constraint_QP(b=info.props, A=info.M, c=info.int_omega)


# %%
# MLE numerics


@jit
def normalize(gamma, int_omega):
    return gamma.reshape(-1, 1) / np.dot(gamma.reshape(-1), int_omega.reshape(-1))


@jit
def T1(gamma0, props, M, int_control):
    gamma = gamma0 * ((M.transpose() @ (props.reshape(-1, 1) / (M @ gamma0))) / int_control)
    return gamma


x1 = FixedPointIteration(fixed_point_fun=T1, jit=True, implicit_diff=True, tol=tol, maxiter=maxiter) \
    .run(np.full_like(info.int_control, 1, dtype=params.dtype) / np.sum(info.int_omega),
         info.props, info.M, info.int_control)


# TODO: for implementation, compute gamma via fix point, threshold, normalize and compute delta, return max(delta)
# as a statistic of the quality of convergence

@jit
def T2(gamma0, props, M, int_control, int_omega):
    pred = np.dot(int_control.reshape(-1), gamma0.reshape(-1))
    denominator = int_control.reshape(-1, 1) + (np.sum(props) - pred) * int_omega.reshape(-1, 1)
    gamma = gamma0 * ((M.transpose() @ (props.reshape(-1, 1) / (M @ gamma0))) / denominator)
    return gamma


x2 = FixedPointIteration(fixed_point_fun=T2, jit=True, implicit_diff=True, tol=tol, maxiter=maxiter) \
    .run(np.full_like(info.int_control, 1, dtype=params.dtype) / np.sum(info.int_omega),
         info.props, info.M, info.int_control, info.int_omega)


@jit
def T3(gamma0, props, M, int_control, int_omega):
    prob = np.dot(int_control.reshape(-1), gamma0.reshape(-1))
    denominator = (int_control.reshape(-1, 1) + (1 - prob) * int_omega.reshape(-1, 1)) * np.sum(props)
    gamma = gamma0 * ((M.transpose() @ (props.reshape(-1, 1) / (M @ gamma0))) / denominator)
    return gamma


x3 = FixedPointIteration(fixed_point_fun=T3, jit=True, implicit_diff=True, tol=tol, maxiter=maxiter) \
    .run(np.full_like(info.int_control, 1, dtype=params.dtype) / np.sum(info.int_omega),
         info.props, info.M, info.int_control, info.int_omega)


@jit
def f(gamma, data):
    M, props, int_control = data
    log_ratio = np.log((M @ gamma.reshape(-1, 1)).reshape(-1) / np.dot(gamma.reshape(-1), int_control.reshape(-1)))
    return -1 * np.sum(props.reshape(-1) * log_ratio)


@jit
def threshold(gamma, tol):
    return gamma * np.array(gamma > tol, dtype=params.dtype)


data = (info.M, info.props, info.int_control)

f(x1[0].reshape(-1), data)
f(x2[0].reshape(-1), data)
f(x3[0].reshape(-1), data)
f(threshold(x3[0].reshape(-1), tol=tol), data)

f(normalize(x1[0].reshape(-1), info.int_omega), data)
f(normalize(x2[0].reshape(-1), info.int_omega), data)
f(normalize(x3[0].reshape(-1), info.int_omega), data)
f(normalize(threshold(x3[0].reshape(-1), tol=tol).reshape(-1), info.int_omega), data)

A = info.M
b = info.props
c = info.int_omega
n_params = A.shape[1]
pg = ProjectedGradient(fun=f,
                       verbose=False,
                       acceleration=True,
                       implicit_diff=True,
                       tol=1e-6,
                       maxiter=2000,
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

pg_sol = pg.run(init_params=a[0].reshape(-1) / np.dot(a[0].reshape(-1), c.reshape(-1)),
                data=(A, b, info.int_control),
                hyperparams_proj=(A_, b_, G, h))
x = pg_sol.params
