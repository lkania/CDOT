from src.load import load
from src.dotdic import DotDic
from experiments.evaluate import _evaluate
from experiments.parameters import build_parameters

# %%
#######################################################
# method arguments
#######################################################

args = DotDic()
args.method = 'bin_mle'
args.k = 20
args.std_signal_region = 3
args.no_signal = False
args.nnls = 'None'
args.data_id = 0
args.cwd = '.'
params = build_parameters(args)

# %%
#######################################################
# load background data
#######################################################
X = load(params.data)
# %%

model = _evaluate(X=X, params=params)

# %%


#

# for real data
# remove entries less than zero
# X_ = X__[np.logical_and(X__ <= 90, X__ > 20)]

# idx = random.choice(params.key, a=X__.shape[0], shape=(250000,))
# X_ = X__[idx]

# %%

# transform data
# from src.transform import transform

#
# trans, tilt_density = transform(X_, c=0.03)
# tX = trans(X_)


# %%

# plot background data
# from importlib import reload
# import src.plot as plot

# reload(plot)

# plot.uniform_histogram(tX, step=0.005)


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
