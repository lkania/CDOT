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


def clear():
    clear_caches()
    clear_backends()


######################################################################
# local libraries
######################################################################
import localize
from src.load import load
from src.dotdic import DotDic
from experiments.parser import parse
from experiments.builder import load_background, load_signal, filter
from src.test.test import test
from src.bin import uniform_bin as _uniform_bin
from src.stat.binom import clopper_pearson

######################################################################

# %%
args = parse()
args.folds = 100
target_data_id = args.data_id

##################################################
# Experiment parameters
##################################################
lambdas = [0.0, 0.001, 0.005, 0.01, 0.05]
quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ks = [5, 10, 15, 20, 25, 30, 35, 40]
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
        pvaluess = []
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

            # save results
            l2.append(l2_)
            pvaluess.append(pvalues)

        idx = np.argmin(np.array(l2))
        selected[quantile].k = ks[idx]
        selected[quantile].pvalues = pvaluess[idx]

        clear()

    return selected


selected = DotDic()
for classifier in params.classifiers:
    args.classifier = classifier
    selected[classifier] = select(args, params, quantiles)

# %%

##################################################
# Power analysis
##################################################
args.data_id = target_data_id

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
                quantile))

            for i in tqdm(range(params.folds), ncols=40):
                pvalue = test(args=args, X=Xs[i]).pvalue
                results[lambda_][quantile].append(pvalue)

            results[lambda_][quantile] = np.array(results[lambda_][quantile])

        clear()

    return results


results = DotDic()
for classifier in params.classifiers:
    args.classifier = classifier
    results[classifier] = test_(args, params, selected, quantiles, lambdas)
