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
from experiments.evaluate import run

# nnls optimizers
from src.opt.scipy import nnls as lawson_scipy_nnls
# from src.opt.lawsonhanson import nnls as lawson_jax_nnls
# from src.opt.jaxopt import nnls as pg_jaxopt_nnls
# from src.opt.pgd import nnls as pg_jax_nnls
# from src.opt.cvx import nnls as conic_cvx_nnls

# methods
from src.background.unbin import mom
# from src.background.bin import chi2 as bin_chi2
from src.background.bin import mle as bin_mle
from src.background.bin import mom as bin_mom

# %%

params = DotDic()
params.seed = 0
params.key = random.PRNGKey(seed=params.seed)
params.k = 30  # high impact on jacobian computation for non-bin methods
params.bins = 44  # high impact on jacobian computation for bin methods
params.data_id = 'real'
params.data = './data/{0}/m_muamu.txt'.format(params.data_id)
params.folds = 200

params.std_signal_region = 3  # amount of contamination
params.no_signal = True  # if simulation is run without signal

# fake signal parameters
params.mu_star = 450
params.sigma_star = 20
params.lambda_star = 0.01

# allow 64 bits
params.dtype = np.float64

#######################################################
# load background data
#######################################################
X__ = load(params.data)

# remove entries less than zero
X_ = X__[np.logical_and(X__ <= 90, X__ > 20)]

# idx = random.choice(params.key, a=X__.shape[0], shape=(250000,))
# X_ = X__[idx]

# %%

# transform data
from src.transform import transform

#
trans, tilt_density = transform(X_, c=0.03)
tX = trans(X_)

# %%


# %%

# plot background data
from importlib import reload
import src.plot as plot

reload(plot)

plot.uniform_histogram(tX, step=0.005)

# %%
models = []
for estimator in [bin_mle]:  # [bin_mle, bin_mom, mom]:
    params.estimator = estimator
    model = run(params=params)
    models.append(model)
print('Finished fitting')

# %%
# names = ['bin_mle', 'bin_mom', 'nob_mom']
# colors = ['red', 'magenta', 'orange']
# for i in np.arange(len(models)):
#     model = models[i]
#     model.name = names[i]
#     model.color = colors[i]
#
# # %%
# # bin_mom should have a lower gamma error than bin_mle
# for model in models:
#     print('{0} \t gamma error: \t{1}'.format(model.name, model.gamma_error))
#
# print('\n')
#
# for model in models:
#     print('{0} \t signal error: \t{1}'.format(model.name, model.signal_error))
#
# # %%
# from importlib import reload
# import src.plot as plot
#
# reload(plot)
#
# # %%
#
# plot.residuals(X=params.trans(params.X),
#                lower=params.trans(params.lower),
#                upper=params.trans(params.upper),
#                methods=models,
#                size=3)
#
# # %%
#
# plot.residuals(X=params.trans(params.X),
#                lower=params.trans(params.lower),
#                upper=params.trans(params.upper),
#                methods=models,
#                step=0.005)
#
# # %%
# plot.density(X=params.trans(params.X),
#              lower=params.trans(params.lower),
#              upper=params.trans(params.upper),
#              methods=models)
#
# # %%
# plot.density(X=params.trans(params.X),
#              lower=params.trans(params.lower),
#              upper=params.trans(params.upper),
#              methods=models,
#              step=0.0005)
