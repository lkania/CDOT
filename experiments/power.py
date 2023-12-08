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

from jax import numpy as np, clear_caches, clear_backends, random
######################################################################
# local libraries
######################################################################
# import localize
from src.load import load
from src.dotdic import DotDic
from experiments.parser import parse
from experiments.builder import load_background, load_signal, filter
from src.test.test import test
from src.bin import uniform_bin as _uniform_bin
from src.stat.binom import clopper_pearson

######################################################################

# [Lucas] Plots for testing
# - test data: fix k, plot p-value distribution through violent plots vs cut-off point of classifier.
# Do one for lambda = 0 and and another one for lambda = 0.05

# DONE:
# - Test data: fix k, plot CDF of p-values for different lambdas.
# - Test data: fix k, plot power at fix alpha-level vs cut-off h (not transformed)
# - Test data: fix k, plot power at fix alpha-level vs cut-off th (transformed)

# %%
args = parse()
args.folds = 100
# args.cwd = '.'

##################################################
# Experiment parameters
##################################################
lambdas = [0.0, 0.001, 0.005, 0.01, 0.05]
quantiles = [0.0, 0.1, 0.2, 0.4, 0.5]
ks = [10, 20, 30]
args.ks = None
args.classifiers = ['tclass', 'class']
##################################################
# Model Selection
##################################################
args.data_id = 'val'
args.lambda_star = 0.0

params = load_background(args)
params = load_signal(args, params)


def select(args, params, quantiles):
    cutoffs = np.quantile(params.background.c[args.classifier],
                          q=np.array(quantiles),
                          axis=0)

    selected = DotDic()
    for c, cutoff in enumerate(cutoffs):
        quantile = quantiles[c]
        selected[quantile] = DotDic()
        selected[quantile].cutoff = cutoff

        args.cutoff = cutoff
        Xs = filter(args, params)

        # run test with background complexity K and compute
        # L2 distance between the CDF of the p-values and
        # the uniform CDF
        l2 = []
        inc = np.square((np.arange(params.folds) + 1) / params.folds)
        for k in ks:

            args.k = k
            print("\nValidation Classifier={0} K={1} cutoff={2}\n".format(
                args.classifier, k, quantile))
            pvalues = []
            for i in tqdm(range(params.folds), ncols=40):
                pvalues.append(test(args=args, X=Xs[i]).pvalue)

            # compute l2 distance proxy
            pvalues = np.array(pvalues)
            pvalues = np.sort(pvalues)
            pvalues_diff = np.concatenate(
                (pvalues[1:], np.array([1]))) - pvalues
            l2_ = np.sum(inc * pvalues_diff)
            l2_ += np.mean(np.square(pvalues)) - 2 / 3
            l2.append(l2_)

        selected[quantile].k = ks[np.argmin(np.array(l2))]
    return selected


selected = DotDic()
for classifier in params.classifiers:
    args.classifier = classifier
    selected[classifier] = select(args, params, quantiles)

# %%

##################################################
# Power analysis
##################################################
args.data_id = '3b'

params = load_background(args)
params = load_signal(args, params)


def test_(args, params, selected, quantiles, lambdas):
    results = DotDic()
    for lambda_ in lambdas:

        # Set true signal proportion
        args.lambda_star = lambda_
        results[lambda_] = DotDic()

        params = load_signal(args, params)

        for quantile in quantiles:

            # Set cutoff point
            cutoff = selected[args.classifier][quantile].cutoff
            args.k = selected[args.classifier][quantile].k
            args.cutoff = cutoff
            results[lambda_][quantile] = []
            Xs = filter(args, params)

            # Run procedures on all datasets
            print("\nTest Classifier={0} K={1} lambda={2} cutoff={3}\n".format(
                args.classifier,
                args.k,
                args.lambda_star,
                quantiles))

            for i in tqdm(range(params.folds), ncols=40):
                pvalue = test(args=args, X=Xs[i]).pvalue
                results[lambda_][quantile].append(pvalue)

            results[lambda_][quantile] = np.array(results[lambda_][quantile])

        # The following lines kill idle threads
        clear_caches()
        clear_backends()

    return results


results = DotDic()
for classifier in params.classifiers:
    args.classifier = classifier
    results[classifier] = test_(args, params, selected, quantiles, lambdas)

# %%
fontsize = 16
colors = distinctipy.get_colors(
    len(lambdas),
    exclude_colors=[(0, 0, 0), (1, 1, 1),
                    (1, 0, 0), (0, 0, 1)],
    rng=0)
row = -1
transparency = 0.5


def plot_series_with_uncertainty(ax, x, mean,
                                 lower=None,
                                 upper=None,
                                 label='',
                                 color='black',
                                 fmt='',
                                 markersize=5,
                                 elinewidth=1,
                                 capsize=3,
                                 set_xticks=True):
    if set_xticks:
        ax.set_xticks(x)

    mean = np.array(mean).reshape(-1)
    yerr = None
    if lower is not None and upper is not None:
        lower = np.array(lower).reshape(-1)
        upper = np.array(upper).reshape(-1)
        l = mean - lower
        u = upper - mean
        yerr = np.vstack((l, u)).reshape(2, -1)

    ax.errorbar(x=x,
                y=mean,
                yerr=yerr,
                color=color,
                capsize=capsize,
                markersize=markersize,
                elinewidth=elinewidth,
                fmt=fmt,
                label=label)


fig, axs = plt.subplots(nrows=2, ncols=2,
                        sharex='none', sharey='none',
                        figsize=(20, 20))
row = -1


###################################################################
# Plot power vs threshold
###################################################################

def plot(ax, results, args, eps=1e-2, alpha=0.05):
    ax.set_title('Clopper-Pearson CI for I(Test=1) at alpha=0.05',
                 fontsize=fontsize)
    ax.set_ylim([0 - eps, 1 + eps])

    ax.axhline(y=alpha, color='red',
               linestyle='-', label='{0}'.format(alpha))
    results = results[args.classifier]
    for i, lambda_ in enumerate(lambdas):
        means = []
        lowers = []
        uppers = []
        for quantile in quantiles:
            pvalues = results[lambda_][quantile]
            tests = np.array(pvalues <= 0.05, dtype=np.int32)
            cp = clopper_pearson(n_successes=[np.sum(tests)],
                                 n_trials=args.folds,
                                 alpha=alpha)[0]
            means.append(np.mean(tests))
            lowers.append(cp[0])
            uppers.append(cp[1])
        plot_series_with_uncertainty(ax, quantiles, means, lowers, uppers,
                                     color=colors[i],
                                     label='Lambda={0}'.format(lambda_))

    ax.legend(fontsize=fontsize)


row += 1
ax = axs[row, 0]
ax.set_xlabel('Threshold (w/o decorrelation)', fontsize=fontsize)
args.classifier = 'class'
plot(ax=ax, results=results, args=args)
ax = axs[row, 1]
ax.set_xlabel('Threshold (with decorrelation)', fontsize=fontsize)
args.classifier = 'tclass'
plot(ax=ax, results=results, args=args)


###################################################################
# Plot distribution and cdf of pvalues
###################################################################


def plot(ax, results, args, quantile, eps=1e-2):
    results = results[args.classifier]
    ax.set_title(
        'CDF of pvalues ({0}% Background reject)'.format(quantile * 100),
        fontsize=fontsize)
    ax.axline([0, 0], [1, 1], color='red')
    ax.set_ylim([0 - eps, 1 + eps])
    ax.set_xlim([0 - eps, 1 + eps])

    for i, lambda_ in enumerate(lambdas):
        pvalues = results[lambda_][quantile]
        sns.ecdfplot(
            data=pvalues,
            hue_norm=(0, 1),
            legend=False,
            ax=ax,
            color=colors[i],
            alpha=1,
            label='Lambda={0}'.format(lambda_))

    ax.legend(fontsize=fontsize)


row += 1
ax = axs[row, 0]
ax.set_xlabel('Threshold (w/o decorrelation)', fontsize=fontsize)
args.classifier = 'class'
plot(ax=ax, results=results, args=args, quantile=0.2)
ax = axs[row, 1]
ax.set_xlabel('Threshold (with decorrelation)', fontsize=fontsize)
args.classifier = 'tclass'
plot(ax=ax, results=results, args=args, quantile=0.2)

###################################################################
# save figure
###################################################################
path = '{0}/summaries/'.format(params.cwd)
Path(params.path).mkdir(parents=True, exist_ok=True)
filename = path + '{0}.pdf'.format(params.data_id)
fig.savefig(fname=filename, bbox_inches='tight')
plt.close(fig)
print('\nSaved to {0}'.format(filename))
