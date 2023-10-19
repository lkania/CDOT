# TODO: Compute p-value distribution
######################################################################
# Utilities
######################################################################
from tqdm import tqdm
from scipy.stats._binomtest import _binom_exact_conf_int as clopper_pearson, \
    _binom_wilson_conf_int as wilson
import statsmodels.api as sm
######################################################################
# Configure matplolib
######################################################################
import distinctipy
import matplotlib

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
from experiments.builder import build_parameters
from src.random import add_signal
from src.ci.delta import delta_ci
from src.bin import proportions, adaptive_bin, full_adaptive_bin
from src.bin import uniform_bin as _uniform_bin

uniform_bin = lambda X, lower, upper, n_bins: _uniform_bin(n_bins=n_bins)

######################################################################

args = parse()
params = build_parameters(args)
ks = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


# %%
def run(params, data, k):
    params.k = k

    method = DotDic()
    method.name = params.name
    method.k = params.k
    method.background.X = data.background.X
    method.X = data.X
    method.signal.X = data.signal.X

    params.background.fit(params=params, method=method)

    def estimate(h_hat):
        lambda_hat0, aux = method.background.estimate_lambda(h_hat)
        return lambda_hat0, (lambda_hat0, aux)

    influence_, aux_ = method.background.influence(estimate)
    lambda_hat0, aux = aux_
    gamma_hat, gamma_aux = aux
    gamma_error, poisson_nll, multinomial_nll = gamma_aux

    # compute confidence interval

    ci, aux = delta_ci(
        point_estimate=np.array([lambda_hat0]),
        influence=influence_)
    std = aux[0]

    # Test
    method.delta_ci = ci[0]
    method.test = not ((method.delta_ci[0] <= 0) and (0 <= method.delta_ci[1]))

    # save results
    method.lambda_hat0 = lambda_hat0
    method.std = std
    method.gamma_hat = gamma_hat
    method.gamma_error = gamma_error
    method.poisson_nll = poisson_nll
    method.multinomial_nll = multinomial_nll

    return method


# %%


# prepare data
datas = []
print("\nPreparing datasets\n")
for i in tqdm(range(params.folds), dynamic_ncols=False):
    data = DotDic()
    X = params.background.X[params.idxs[i]]
    add_signal(X=X, params=params, method=data)
    datas.append(data)

# k-dependent code
results = DotDic()
for k in ks:
    print("\nTesting K={0}\n".format(k))
    results[k] = []
    for i in tqdm(range(params.folds), dynamic_ncols=False):
        results[k].append(run(params, datas[i], k))

# %%
# TODO: save results, just in case I need to modify the plots

# %%
fontsize = 16
colors = distinctipy.get_colors(len(ks))
alpha = 0.05
row = 0
transparency = 0.5


def predictions(from_, to_, method):
    basis = params.basis.integrate(method.k, from_, to_)
    return (basis @ method.gamma_hat).reshape(-1)


def plot_series_with_uncertainty(ax, x, mean,
                                 lower=None,
                                 upper=None,
                                 label='',
                                 color='black', fmt='',
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
        yerr = np.vstack((mean - lower, upper - mean)).reshape(2, -1)

    ax.errorbar(x=x,
                y=mean,
                yerr=yerr,
                color=color,
                capsize=capsize,
                markersize=markersize,
                elinewidth=elinewidth,
                fmt=fmt,
                label=label)


def plot_hist_with_uncertainty(ax,
                               from_,
                               to_,
                               mean,
                               lower=None,
                               upper=None,
                               jitter=0,
                               color='black',
                               label=''):
    bin_centers = jitter + (from_ + to_) / 2
    plot_series_with_uncertainty(ax=ax,
                                 x=bin_centers,
                                 mean=mean,
                                 lower=lower,
                                 upper=upper,
                                 color=color,
                                 label=label,
                                 set_xticks=False,
                                 markersize=2,
                                 elinewidth=1,
                                 capsize=1,
                                 fmt='o')


def plot_hists(ax, binning, ms):
    from_, to_ = binning(
        X=ms[0].tX,
        lower=params.tlower,
        upper=params.tupper,
        n_bins=params.bins)
    props = proportions(X=ms[0].tX, from_=from_, to_=to_)[0]
    # TODO: replace by clopper-pearson confidence intervals
    stds = np.sqrt(props * (1 - props) / ms[0].tX.shape[0])
    ax.set_xlabel('Mass (projected scale)', fontsize=fontsize)
    ax.set_ylabel('Normalized counts', fontsize=fontsize)
    ax.axvline(x=params.tlower,
               color='red', linestyle='--')
    ax.axvline(x=params.tupper,
               color='red', linestyle='--',
               label='Signal region')
    plot_hist_with_uncertainty(ax=ax,
                               from_=from_,
                               to_=to_,
                               mean=props,
                               lower=props - stds,
                               upper=props + stds,
                               color='black',
                               label='Data')
    for i, m in enumerate(ms):
        plot_hist_with_uncertainty(
            ax=ax,
            from_=from_,
            to_=to_,
            mean=predictions(from_, to_, m),
            jitter=(i + 1) * 1e-1 * (to_ - from_),
            color=colors[i],
            label='K={0}'.format(m.k))

    ax.legend(fontsize=fontsize)


def plot_hists_with_uncertainty(ax, binning):
    ax.set_xlabel('Mass (projected scale)', fontsize=fontsize)
    ax.set_ylabel('Normalized counts', fontsize=fontsize)
    ax.axvline(x=params.tlower,
               color='red', linestyle='--')
    ax.axvline(x=params.tupper,
               color='red', linestyle='--',
               label='Signal region')
    tX = params.trans(params.mixture.X)
    from_, to_ = binning(X=tX,
                         lower=params.tlower,
                         upper=params.tupper,
                         n_bins=params.bins)

    props = proportions(X=tX, from_=from_, to_=to_)[0]
    stds = np.sqrt(props * (1 - props) / params.mixture.n)
    plot_hist_with_uncertainty(ax=ax,
                               from_=from_,
                               to_=to_,
                               mean=props,
                               lower=props - stds,
                               upper=props + stds,
                               color='black',
                               label='Data')

    for i, k in enumerate(ks):
        preds = [predictions(from_, to_, results[k][l])
                 for l in range(params.folds)]
        preds = np.array(preds)
        mean = np.mean(preds, axis=0)
        lower = np.quantile(preds, q=alpha / 2, axis=0)
        upper = np.quantile(preds, q=1 - alpha / 2, axis=0)
        plot_hist_with_uncertainty(
            ax=ax,
            from_=from_,
            to_=to_,
            mean=mean,
            lower=lower,
            upper=upper,
            jitter=(i + 1) * 1e-1 * (to_ - from_),
            color=colors[i],
            label='K={0}'.format(k))
    ax.legend(fontsize=fontsize)


fig, axs = plt.subplots(5, 3,
                        sharex='none', sharey='none',
                        figsize=(50, 50))

##################################################################
# Instance fit
###################################################################
ax = axs[row, 0]
instance = [results[k][0] for k in ks]
ax.set_title('Adaptive binning (Control Region) (One instance)',
             fontsize=fontsize)
plot_hists(ax, adaptive_bin, instance)

ax = axs[row, 1]
ax.set_title('Adaptive binning (One instance)', fontsize=fontsize)
plot_hists(ax, full_adaptive_bin, instance)

ax = axs[row, 2]
ax.set_title('Uniform binning (One instance)', fontsize=fontsize)
plot_hists(ax, uniform_bin, instance)

##################################################################
# Average fit
###################################################################
row += 1

ax = axs[row, 0]
ax.set_title('Adaptive binning (Control Region) (Average)', fontsize=fontsize)
plot_hists_with_uncertainty(ax, adaptive_bin)

ax = axs[row, 1]
ax.set_title('Adaptive binning (Average)', fontsize=fontsize)
plot_hists_with_uncertainty(ax, full_adaptive_bin)

ax = axs[row, 2]
ax.set_title('Uniform binning (Average)', fontsize=fontsize)
plot_hists_with_uncertainty(ax, uniform_bin)

###################################################################
# Plot estimates histogram
###################################################################
row += 1
ax = axs[row, 0]
ax.set_title('Score distribution', fontsize=fontsize)
ax2 = axs[row, 1]
ax2.set_title('QQ-Plot of scores', fontsize=fontsize)
means = []
lowers = []
uppers = []
for i, k in enumerate(ks):
    estimates = [results[k][l].lambda_hat0 for l in range(params.folds)]
    estimates = np.array(estimates)
    bias = estimates - params.lambda_star
    means.append(np.mean(bias))
    lowers.append(np.quantile(bias, q=alpha / 2))
    uppers.append(np.quantile(bias, q=1 - alpha / 2))
    scores = [
        (results[k][l].lambda_hat0 - params.lambda_star) / results[k][l].std
        for l in range(params.folds)]
    scores = np.array(scores)
    ax.hist(scores,
            alpha=1,
            bins=30,
            density=True,
            histtype='step',
            label='K={0}'.format(k),
            color=colors[i])

    # sm.qqplot(scores, line='45', ax=ax2, a=0, loc=0, scale=1)
    pp = sm.ProbPlot(scores, fit=False, a=0, loc=0, scale=1)
    qq = pp.qqplot(ax=ax2,
                   marker='.',
                   label='K={0}'.format(k),
                   markerfacecolor=colors[i],
                   markeredgecolor=colors[i],
                   alpha=1)

sm.qqline(ax=ax2, line='45', fmt='r--')
ax.axvline(x=0, color='red', linestyle='-')
ax.legend(fontsize=fontsize)
ax2.legend(fontsize=fontsize)
###################################################################
# Plot estimate bias
###################################################################
ax = axs[row, 2]
ax.set_title('Empirical bias (lambda_hat - lambda_star)', fontsize=fontsize)
ax.axhline(y=0, color='red', linestyle='-')
plot_series_with_uncertainty(ax, ks, means, lowers, uppers)

###################################################################
# Plot P(tests = 1 | under H_0)
###################################################################
row += 1
ax = axs[row, 0]
ax.set_title('Clopper-Pearson CI for I(Test=1)', fontsize=fontsize)
means = []
lowers = []
uppers = []
for i, k in enumerate(ks):
    tests = [results[k][l].test for l in range(params.folds)]
    tests = np.array(tests, dtype=np.int32)
    cp = clopper_pearson(k=np.sum(tests),  # number of successes
                         n=params.folds,  # number of trials
                         alternative='two-sided',
                         confidence_level=1 - alpha)
    means.append(np.mean(tests))
    lowers.append(np.mean(cp[0]))
    uppers.append(np.mean(cp[1]))

ax.set_ylim([0, 1])
ax.axhline(y=alpha, color='red', linestyle='-', label='{0}'.format(alpha))
plot_series_with_uncertainty(ax, ks, means, lowers, uppers)

###################################################################
# Optimization statistics
###################################################################
row += 1


def plot_stat(ax, title, stat):
    ax.set_title(title, fontsize=fontsize)
    means = []
    lowers = []
    uppers = []
    for i, k in enumerate(ks):
        stats = [results[k][l][stat] for l in range(params.folds)]
        stats = np.array(stats)
        means.append(np.mean(stats))
        lowers.append(np.quantile(stats, q=alpha / 2))
        uppers.append(np.quantile(stats, q=1 - alpha / 2))
    plot_series_with_uncertainty(ax, ks, means, lowers, uppers)


plot_stat(axs[row, 0], 'Gamma error (delta update)', 'gamma_error')
plot_stat(axs[row, 1], 'Multinomial negative-ll', 'multinomial_nll')
plot_stat(axs[row, 2], 'Poisson negative-nll', 'poisson_nll')
###################################################################
# save figure
###################################################################
path = '{0}/summaries/testing/{1}/{2}/{3}/'.format(
    params.cwd,
    params.data_id,
    params.method,
    params.optimizer)
Path(path).mkdir(parents=True, exist_ok=True)
filename = path + '{0}.pdf'.format(params.lambda_star)
fig.savefig(fname=filename,
            bbox_inches='tight')
plt.close(fig)
print('\nSaved to {0}'.format(filename))
