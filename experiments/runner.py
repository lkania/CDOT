# TODO: in order to easily support BCA, compute the necessary
# resambles and jacnife statistics and then execute the
# rest of the function bcajack in bcaboot

# In order to take advantage of jit, run each method independently
# in this way we avoid recompilation of common methods

import localize
from experiments.evaluate import run_and_save
from src.dotdic import DotDic
from argparse import ArgumentParser

# nnls optimizers
# from src.opt.scipy import nnls as lawson_scipy_nnls
# from src.opt.lawsonhanson import nnls as lawson_jax_nnls
# from src.opt.jaxopt import nnls as pg_jaxopt_nnls
# from src.opt.pgd import nnls as pg_jax_nnls
# from src.opt.cvx import nnls as conic_cvx_nnls

# methods
from src.background.unbin import mom
from src.background.bin import chi2 as bin_chi2
from src.background.bin import mle as bin_mle
from src.background.bin import mom as bin_mom

#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as np
import jax.random as random

#######################################################

parser = ArgumentParser()
parser.add_argument("--method", type=str, default='bin_mle')
parser.add_argument("--k", type=int, default=30)
parser.add_argument("--std_signal_region", type=int, default=3)
parser.add_argument("--no_signal", type=str, default='False')
parser.add_argument("--nnls", default='None', type=str)
parser.add_argument("--data_id", default=50, type=int)
args = parser.parse_args()
args.no_signal = (args.no_signal.lower() == 'true')

#######################################################
# Simulation parameters
#######################################################

params = DotDic()
params.seed = 0
params.key = random.PRNGKey(seed=params.seed)
params.k = args.k  # high impact on jacobian computation for non-bin methods
params.bins = 100  # high impact on jacobian computation for bin methods
params.data_id = args.data_id
params.data = '../data/{0}/m_muamu.txt'.format(params.data_id)
params.folds = 200

params.std_signal_region = args.std_signal_region  # amount of contamination
params.no_signal = args.no_signal  # if simulation is run without signal

# fake signal parameters
params.mu_star = 450
params.sigma_star = 20
params.lambda_star = 0.01

# allow 64 bits
params.dtype = np.float64

# numerical methods
params.tol = 1e-6
params.maxiter = 10000

match args.method:
    case 'bin_mle':
        params.estimator = bin_mle
    case 'bin_mom':
        params.estimator = bin_mom
    case 'bin_chi2':
        params.estimator = bin_chi2
    case 'mom':
        params.estimator = mom

# match args.nnls:
#     case 'conic_cvx':
#         params.nnls = conic_cvx_nnls
#     case 'pg_jaxopt':
#         params.nnls = pg_jaxopt_nnls
#     case 'pg_jax':
#         params.nnls = pg_jax_nnls
#     case 'lawson_scipy':
#         params.nnls = lawson_scipy_nnls
#     case 'lawson_jax':
#         params.nnls = lawson_jax_nnls

# the name should contain the following information
# - optimizer
# - number of background parameters
# - dataset id
# - std for signal region
# - presence of signal
if args.nnls == 'None':
    params.name = args.method
else:
    params.name = '_'.join([args.method, args.nnls])

params.name = '_'.join(
    [params.name,
     str(params.k),
     str(params.std_signal_region),
     str(params.no_signal),
     str(params.data_id),
     str(params.folds)])

# sanity checks
assert (params.bins > params.k + 1)

print('Name of the method: {0}'.format(params.name))
run_and_save(params=params)
print('{0} finished'.format(params.name))
