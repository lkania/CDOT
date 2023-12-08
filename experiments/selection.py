######################################################################
# Utilities
######################################################################
from tqdm import tqdm
import statsmodels.api as sm
######################################################################
# Configure matplolib
######################################################################
import distinctipy
import matplotlib
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 600
from pathlib import Path
#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np
######################################################################
# local libraries
######################################################################
import localize
from src.dotdic import DotDic
from experiments.parser import parse
from experiments.builder import build
from src.test.test import test
from src.bin import proportions, adaptive_bin, full_adaptive_bin
from src.bin import uniform_bin as _uniform_bin
from src.stat.binom import clopper_pearson

uniform_bin = lambda X, lower, upper, n_bins: _uniform_bin(n_bins=n_bins)

######################################################################

# [Lucas] Plots for testing
# - Validation: CDF of p-value vs K (complexity of background)
# Three plots, lambda 0, lambda 0.05, lambda 0.1


print("\nModel selection experiment\n")
ks = [10, 15, 20, 25, 30, 35, 40, 45, 50]

args = parse()
args.folds = 1000
args.data_id = 'val'
args.k = 5
args.ks = None
args.lambda_star = 0.0
args.cutoff = 0.5
params = build(args=args)

results = DotDic()
for k in ks:
    args.k = k
    print("\nTesting K={0}\n".format(k))
    results[k] = []
    for i in tqdm(range(params.folds), ncols=40):
        results[k].append(test(args=args, X=params.Xs[i]))

# %%
fontsize = 16
colors = distinctipy.get_colors(
    n_colors=len(ks),
    exclude_colors=[(0, 0, 0), (1, 1, 1),
                    (1, 0, 0), (0, 0, 1)],
    rng=0)
alpha = 0.05
row = -1
transparency = 0.5
eps = 1e-2

fig, axs = plt.subplots(nrows=1, ncols=1,
                        sharex='none', sharey='none',
                        figsize=(10, 10))
row = -1

###################################################################
# Plot power vs threshold
###################################################################
row += 1
ax = axs
ax.set_title('CDF of pvalues (cutoff={0})'.format(args.cutoff),
             fontsize=fontsize)
ax.axline([0, 0], [1, 1], color='red')
ax.set_ylim([0 - eps, 1 + eps])
ax.set_xlim([0 - eps, 1 + eps])

for i, k in enumerate(ks):
    pvalues = np.array([results[k][l].pvalue for l in range(params.folds)])

    sns.ecdfplot(
        data=pvalues,
        hue_norm=(0, 1),
        legend=False,
        ax=ax,
        color=colors[i],
        alpha=1,
        label='K={0}'.format(k) if k is not None else 'Model selection')

ax.legend(fontsize=fontsize)

###################################################################
# save figure
###################################################################
Path(params.path).mkdir(parents=True, exist_ok=True)
filename = params.path + 'selection.pdf'
fig.savefig(fname=filename,
            bbox_inches='tight')
plt.close(fig)
print('\nSaved to {0}'.format(filename))
