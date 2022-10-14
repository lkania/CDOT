#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as np
import jax.random as random
#######################################################

from src.load import load
from src.dotdic import DotDic

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
params.maxiter = 5000

# %%

#######################################################
# load background data
#######################################################
X = load(params.data)

# %%


# transform data
from src.transform import transform

trans, tilt_density = transform(X)
tX = trans(X)
lower = trans(450 - 3 * 20)
upper = trans(450 + 3 * 20)

# # %%
# # least squares numerics
# from src.background.bin.preprocess import preprocess
#
# info = preprocess(X=tX, lower=lower, upper=upper, params=params)
#
# # %%
# # non-negative least squares numerics
#
# from src.normalize import normalize
# from src.opt.jaxopt import nnls as pg_jaxopt_nnls
# from src.opt.jaxopt import nnls_with_linear_constraint
#
# x1 = normalize(gamma_and_aux=pg_jaxopt_nnls(b=info.props, A=info.M), int_omega=info.int_omega)
#
# x2 = nnls_with_linear_constraint(b=info.props, A=info.M, c=info.int_omega)
#
# x3 = nnls_with_linear_constraint_QP(b=info.props, A=info.M, c=info.int_omega)

# %%
# MLE numerics
import src.background.bin.mle as bin_mle

info = bin_mle.fit(X=tX, lower=lower, upper=upper, params=params)

# %%

x = info.compute_gamma(info.props)
