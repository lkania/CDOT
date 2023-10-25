# TODO: Compute p-value distribution
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
import numpy as onp
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
from src.stat.binom import clopper_pearson

uniform_bin = lambda X, lower, upper, n_bins: _uniform_bin(n_bins=n_bins)

######################################################################

args = parse()
params = build_parameters(args)
ks = [None] + params.ks
# for plotting
ks_ = [params.ks[0] - 1 if v is None else v for v in ks]


# %%
def run(params, data, k):
    params.k = k

    method = DotDic()
    method.background = DotDic()
    method.signal = DotDic()
    method.model_selection = DotDic()
    method.k = params.k
    method.background.X = data.background.X
    method.X = data.X

    if method.k is None:
        method.model_selection.activated = True
        val_error = onp.zeros(len(params.ks))
        for i, k in enumerate(params.ks):
            method.k = k
            params.background.fit(params=params, method=method)
            val_error[i] = method.background.validation_error()
        k_star = onp.argmin(val_error)
        method.model_selection.val_error = val_error
        params.k = params.ks[k_star]
        method.k = params.k

    method.model_selection.activated = False
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
for i in tqdm(range(params.folds), ncols=40):
    data = DotDic()
    data.background = DotDic()
    X = params.background.X[params.idxs[i]]
    add_signal(X=X, params=params, method=data)
    datas.append(data)

# k-dependent code
results = DotDic()
for k in ks:
    print("\nTesting K={0}\n".format(k))
    results[k] = []
    for i in tqdm(range(params.folds), ncols=40):
        results[k].append(run(params, datas[i], k))

# %%
fontsize = 16
colors = distinctipy.get_colors(len(ks),
                                exclude_colors=[(0, 0, 0), (1, 1, 1),
                                                (1, 0, 0), (0, 0, 1)],
                                rng=0)
alpha = 0.05
row = -1
transparency = 0.1


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


def plot_hists(ax, binning, ks, ms):
    from_, to_ = binning(
        X=ms[0].tX,
        lower=params.tlower,
        upper=params.tupper,
        n_bins=params.bins)

    props = proportions(X=ms[0].tX, from_=from_, to_=to_)[0]
    counts = props * ms[0].tX.shape[0]

    cis = clopper_pearson(n_successes=counts,
                          n_trials=ms[0].tX.shape[0],
                          alpha=alpha)

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
                               lower=cis[:, 0].reshape(-1),
                               upper=cis[:, 1].reshape(-1),
                               color='black',
                               label='Data (Clopper-Pearson CI)')
    for i, k in enumerate(ks):
        m = ms[i]
        label = 'Model selection' if k is None else 'K={0}'.format(k)
        plot_hist_with_uncertainty(
            ax=ax,
            from_=from_,
            to_=to_,
            mean=params.basis.predict(
                gamma=m.gamma_hat,
                k=m.k,
                from_=from_,
                to_=to_).reshape(-1),
            jitter=(i + 1) * 1e-1 * (to_ - from_),
            color=colors[i],
            label=label)

        if k is None:
            ax.axvline(x=m.model_selection.from_,
                       color='blue', linestyle='--')
            ax.axvline(x=m.model_selection.to_,
                       color='blue', linestyle='--',
                       label='Model selection region')

    ax.legend(fontsize=fontsize)


def plot_hists_with_uncertainty(ax, binning, ks):
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
    counts = props * tX.shape[0]
    cis = clopper_pearson(n_successes=counts,
                          n_trials=tX.shape[0],
                          alpha=alpha)

    plot_hist_with_uncertainty(ax=ax,
                               from_=from_,
                               to_=to_,
                               mean=props,
                               lower=cis[:, 0].reshape(-1),
                               upper=cis[:, 1].reshape(-1),
                               color='black',
                               label='Data (Clopper-Pearson CI)')

    for i, k in enumerate(ks):
        preds = [params.basis.predict(
            gamma=results[k][l].gamma_hat,
            k=results[k][l].k,
            from_=from_,
            to_=to_).reshape(-1) for l in range(params.folds)]
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
            label='K={0}'.format(k) if k is not None else 'Model selection')
        if k is None:
            froms = np.array([results[k][l].model_selection.from_
                              for l in range(params.folds)])
            tos = np.array([results[k][l].model_selection.to_
                            for l in range(params.folds)])
            ax.axvline(x=np.mean(froms),
                       color='blue', linestyle='--')
            ax.axvline(x=np.mean(tos),
                       color='blue', linestyle='--',
                       label='Average model selection region')
    ax.legend(fontsize=fontsize)


def plot_stat(ax, title, stat):
    ax.set_title(title, fontsize=fontsize)
    means = []
    lowers = []
    uppers = []
    for i, k in enumerate(ks):
        stats = [stat(results[k][l]) for l in range(params.folds)]
        stats = np.array(stats)
        means.append(np.mean(stats))
        lowers.append(np.quantile(stats, q=alpha / 2))
        uppers.append(np.quantile(stats, q=1 - alpha / 2))
    plot_series_with_uncertainty(ax, ks_, means, lowers, uppers)


fig, axs = plt.subplots(5, 3,
                        sharex='none', sharey='none',
                        figsize=(50, 50))

##################################################################
# Find top 3 performers
# Currently, I choose them based on average bias
###################################################################
means = []
for i, k in enumerate(params.ks):
    estimates = [results[k][l].lambda_hat0 for l in range(params.folds)]
    estimates = np.array(estimates)
    bias = estimates - params.lambda_star
    means.append(np.mean(bias))
means = np.abs(np.array(means))
selected = (np.array(params.ks)[(np.argsort(means)[:3])]).tolist()
top_performers = [None] + selected

##################################################################
# Instance fit
###################################################################
row += 1
ax = axs[row, 0]
instance = [results[k][0] for k in top_performers]
ax.set_title('Adaptive binning (Control Region) (One instance)',
             fontsize=fontsize)
plot_hists(ax, adaptive_bin, ks=top_performers, ms=instance)

ax = axs[row, 1]
ax.set_title('Adaptive binning (One instance)', fontsize=fontsize)
plot_hists(ax, full_adaptive_bin, ks=top_performers, ms=instance)

ax = axs[row, 2]
ax.set_title('Uniform binning (One instance)', fontsize=fontsize)
plot_hists(ax, uniform_bin, ks=top_performers, ms=instance)

##################################################################
# Average fit
###################################################################
row += 1

ax = axs[row, 0]
ax.set_title('Adaptive binning (Control Region) (Average)', fontsize=fontsize)
plot_hists_with_uncertainty(ax, adaptive_bin, ks=top_performers)

ax = axs[row, 1]
ax.set_title('Adaptive binning (Average)', fontsize=fontsize)
plot_hists_with_uncertainty(ax, full_adaptive_bin, ks=top_performers)

ax = axs[row, 2]
ax.set_title('Uniform binning (Average)', fontsize=fontsize)
plot_hists_with_uncertainty(ax, uniform_bin, ks=top_performers)

###################################################################
# Plot estimates histogram
###################################################################
row += 1
ax = axs[row, 0]
ax.set_title('Score distribution', fontsize=fontsize)
ax2 = axs[row, 1]
ax2.set_title('QQ-Plot of scores', fontsize=fontsize)
for i, k in enumerate(top_performers):
    scores = [
        (results[k][l].lambda_hat0 - params.lambda_star) / results[k][l].std
        for l in range(params.folds)]
    scores = np.array(scores)
    ax.hist(scores,
            alpha=1,
            bins=30,
            density=True,
            histtype='step',
            label='K={0}'.format(k) if k is not None else 'Model selection',
            color=colors[i])

    # sm.qqplot(scores, line='45', ax=ax2, a=0, loc=0, scale=1)
    pp = sm.ProbPlot(scores, fit=False, a=0, loc=0, scale=1)
    qq = pp.qqplot(ax=ax2,
                   marker='.',
                   label='K={0}'.format(
                       k) if k is not None else 'Model selection',
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
ax.axhline(y=0, color='red', linestyle='-')
plot_stat(ax,
          'Empirical bias (lambda_hat - lambda_star)',
          lambda m: m.lambda_hat0 - params.lambda_star)

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
    cp = clopper_pearson(n_successes=[np.sum(tests)],
                         n_trials=params.folds,
                         alpha=alpha)[0]
    means.append(np.mean(tests))
    lowers.append(cp[0])
    uppers.append(cp[1])

ax.set_ylim([0, 1])
ax.axhline(y=alpha, color='red', linestyle='-', label='{0}'.format(alpha))
plot_series_with_uncertainty(ax, ks_, means,
                             lowers, uppers)

###################################################################
# Model selection
###################################################################
vals = [results[None][l].model_selection.val_error for l in range(params.folds)]
vals = np.array(vals)  # folds x dim(ks)
means = np.mean(vals, axis=0)
lowers = np.quantile(vals, q=alpha / 2, axis=0)
uppers = np.quantile(vals, q=1 - alpha / 2, axis=0)
k_star = [results[None][l].k for l in range(params.folds)]

ax = axs[row, 1]
ax.set_title('Model selection (loss)'.format(params.folds),
             fontsize=fontsize)
plot_series_with_uncertainty(ax, params.ks, means, lowers, uppers)
ax = axs[row, 2]
ax.set_title('Model selection (choice distribution)'.format(params.folds),
             fontsize=fontsize)
ax.hist(k_star,
        alpha=1,
        bins=np.concatenate((np.array(params.ks) - 0.5,
                             np.array([params.ks[-1]]))),
        density=False,
        label='Selection',
        color='black')
ax.set_xticks(params.ks)

###################################################################
# Optimization statistics
###################################################################

# plot_stat(axs[row, 2], 'Proportion of zeros in gamma',
#           lambda m: np.mean(m.gamma_hat < params.tol))
row += 1
plot_stat(axs[row, 0],
          'Fix-point error ({0} optimizer)'.format(params.optimizer),
          lambda m: m.gamma_error)
plot_stat(axs[row, 1], 'Multinomial negative-ll',
          lambda m: m.multinomial_nll)
plot_stat(axs[row, 2], 'Poisson negative-nll',
          lambda m: m.poisson_nll)
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
