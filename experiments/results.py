# %%

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 600
from experiments.parameters import PARAMETERS, CIS_DELTA, ESTIMATORS, ERRORS
import jax.random as random
from scipy.stats import norm, probplot
from src.stat.binom import clopper_pearson
from src.dotdic import DotDic
from experiments.parameters import normal
import src.basis.bernstein as basis
from src.transform import transform
from pathlib import Path
from src.bin import proportions, uniform_bin
# import os
from tqdm import tqdm
from functools import partial


# os.chdir('./experiments/')  # when working locally


# %%

# TODO: create plots for the predictions of each model
# TODO: plot metrics of diff model classes (mle vs bin_mom) in the same plot (e.g. cov and bias)


# %%


def load(id, name, data):
    file = np.load('./experiments/results/{0}/{1}.npz'.format(id, name))
    m = DotDic()
    m.name = name

    m.gamma = file['gamma']

    m.bias = file['bias']
    m.est = file['estimates']

    m.cov = file['coverage']
    m.width = file['width']
    m.cis = file['cis']

    m.lower = np.float64(file['lower'])
    m.tlower = data.trans(m.lower)
    m.upper = np.float64(file['upper'])
    m.tupper = data.trans(m.upper)

    m.std_signal_region = int(file['std_signal_region'])
    m.k = int(file['k'])
    m.bins = int(file['bins'])

    m.mu_star = np.float64(file['mu_star'])
    m.sigma_star = np.float64(file['sigma_star'])
    m.sigma2_star = np.float64(file['sigma2_star'])
    m.lambda_star = np.float64(file['lambda_star'])
    m.parameters_star = np.array(
        [m.lambda_star, m.lambda_star, m.mu_star, m.sigma2_star], dtype=np.float64)

    m.gamma_error = file['gamma_error']
    m.signal_error = file['signal_error']

    ##################################################################################################
    # compute:
    # - signal densities
    # - background predictions and residuals
    ##################################################################################################

    # signal density
    n_pos = 100
    m.signal.x_axis = np.linspace(m.mu_star - 3.5 * m.sigma_star, m.mu_star + 3.5 * m.sigma_star, n_pos)
    m.signal.star = m.lambda_star * normal(m.signal.x_axis, mu=m.mu_star, sigma2=m.sigma2_star)
    m.signal.predictions = np.zeros((m.est.shape[0], n_pos))
    m.signal.residuals = np.zeros((m.est.shape[0], n_pos))

    # background
    m.background.x_axis = data.background.x_axis.reshape(-1)
    m.background.star = data.background.empirical_probabilities.reshape(-1)
    m.background.from_ = data.background.from_.reshape(-1)
    m.background.to_ = data.background.to_.reshape(-1)
    m.background.residuals = np.zeros((m.gamma.shape[0], m.background.star.shape[0]))
    m.background.predictions = np.zeros((m.gamma.shape[0], m.background.star.shape[0]))
    basis_ = basis.integrate(k=m.k, a=data.background.from_, b=data.background.to_)

    for i in np.arange(m.est.shape[0]):
        gamma = m.gamma[i, :]
        lambda_ = m.est[i, 1]
        mu = m.est[i, 2]
        sigma2 = m.est[i, 3]

        # estimated density
        m.signal.predictions[i, :] = lambda_ * normal(X=m.signal.x_axis, mu=mu, sigma2=sigma2)
        m.signal.residuals[i, :] = m.signal.predictions[i, :] - m.signal.star

        # bin background predictions
        m.background.predictions[i, :] = (basis_ @ gamma.reshape(-1, 1)).reshape(-1)
        m.background.residuals[i, :] = m.background.predictions[i, :] - m.background.star

    # l2 error computation
    # M = basis.integrated_inner_product(m.k)  # TODO: check
    #
    # m.alt_chi2_background = np.zeros(m.gamma.shape[0])
    # m.chi2_background = np.zeros(m.gamma.shape[0])

    # m.l2_background = np.zeros(m.gamma.shape[0])
    #
    # m.l2_signal = np.zeros(m.gamma.shape[0])

    # error and chi2 computation
    # props_back = props_back.reshape(-1)
    # basis_ = basis.integrate(k=m.k, a=from_, b=to_)
    # #
    # for i in np.arange(m.gamma.shape[0]):
    #     gamma = m.gamma[i, :]
    #     lambda_ = m.est[i, 1]
    #     mu = m.est[i, 2]
    #     sigma = m.est[i, 3]
    #
    #     #     # TODO: compute l2 error for the mixture
    #     #     # TODO: compute the l2 error in the original scale of the data
    #     #     # l2 error in background
    #     #     avg = np.mean((basis.evaluate(k=m.k, X=X) @ gamma.reshape(-1, 1)).reshape(-1))
    #     #     m.l2_background[i] = gamma.reshape((1, -1)) @ M @ gamma.reshape((-1, 1)) - 2 * avg
    #     #
    #     #     # l2 error in signal (gaussian)
    #     #     sigma2_tilde = np.square(sigma) + np.square(m.sigma_star)
    #     #     m.l2_signal[i] = 1 / (2 * np.sqrt(np.pi)) \
    #     #                      * (1 / sigma + 1 / m.sigma_star) \
    #     #                      - 2 * normal(X=mu, mu=m.mu_star, sigma2=sigma2_tilde)
    #     #
    #     #     # residuals and chi2 for background
    #     preds_back = (basis_ @ gamma.reshape(-1, 1)).reshape(-1)
    #     errors_back = (preds_back - props_back)
    #     # chi2_back = np.sum(np.square(errors_back) / preds_back)
    #     #     alt_chi2_back = np.sum(np.square(errors_back) / props_back)
    #     #
    #     #     m.chi2_background[i] = chi2_back
    #     #     m.alt_chi2_background[i] = alt_chi2_back
    #     m.residual_background[i, :] = errors_back
    #     m.preds_background[i, :] = preds_back
    #
    #     # residuals and chi2 for background and signal
    #
    #     # preds_backsig = (1-lambda_) *  (basis_ @ gamma.reshape(-1, 1)).reshape(-1) + lambda_ *
    #     # errors_backsig = (props_backsig - preds_backsig)
    #     #
    #     # m.residual_backsig[i, :] = errors_back
    #     # m.preds_backsig[i, :] = preds_back
    #
    # # print('{0} loaded'.format(m.name))
    return m


def create_hierarchy(df):
    df = df.set_index(['method', 'type']).unstack()
    df.columns = pd.MultiIndex.from_product(df.columns.levels)
    return df


# %%

# load background data and fake signal
data = DotDic()
data.background.X = np.array(np.loadtxt('./data/1/m_muamu.txt'))
trans, tilt_density, inv_trans = transform(base=np.min(data.background.X), c=0.003)
data.trans = trans
from_, to_ = uniform_bin(step=0.01)

# axis in original scale
# data.background.from_ = inv_trans(from_)
# data.background.to_ = np.array(inv_trans(to_).reshape(-1), dtype=np.float64)
# data.background.to_[-1] = np.max(data.background.X)  # change infinity for maximum value
# data.background.x_axis = inv_trans((from_ + to_) / 2)
data.background.from_ = from_
data.background.to_ = to_
data.background.x_axis = (from_ + to_) / 2

data.background.empirical_probabilities, _ = proportions(X=trans(data.background.X), from_=from_, to_=to_)

# seed = 0
# sigma_star = 20
# mu_star = 450
# lambda_star = 0.01
# n = X.shape[0]
# n_signal = np.int32(n * lambda_star)
# key = random.PRNGKey(seed=seed)
# signal = mu_star + sigma_star * random.normal(key, shape=(n_signal,))
# X_with_signal = np.concatenate((X, signal))
# tX_with_signal = trans(X_with_signal)
# props_back_and_sig = proportions(X=tX_with_signal, from_=from_, to_=to_)

print("Data loaded")

# %%

estimators = []
increasing_parameters_bin_mle = [
    # 'bin_mle_2_3_False_50_200',
    # 'bin_mle_3_3_False_50_200',
    'bin_mle_4_3_False_50_200',
    'bin_mle_5_3_False_50_200',
    'bin_mle_10_3_False_50_200',
    'bin_mle_20_3_False_50_200',
    'bin_mle_30_3_False_50_200'
]
estimators += increasing_parameters_bin_mle
# increasing_parameters_bin_mom = [
#     'bin_mom_5_3_False_50_200',
#     'bin_mom_10_3_False_50_200',
#     # 'bin_mom_20_3_False_50_200',
#     'bin_mom_30_3_False_50_200',
#     'bin_mom_40_3_False_50_200'
# ]
# estimators += increasing_parameters_bin_mom
# increasing_contamination_1 = [
#     'bin_mle_30_3_False_50_200',
#     'bin_mle_30_2_False_50_200',
#     'bin_mle_30_1_False_50_200']
# estimators += increasing_contamination_1
# increasing_sample_1 = [
#     'bin_mle_30_3_False_5_200',
#     'bin_mle_30_3_False_50_200']
# estimators += increasing_sample_1
# increasing_sample_2 = [
#     'bin_mom_lawson_scipy_30_3_False_5_200',
#     'bin_mom_lawson_scipy_30_3_False_50_200']
# estimators += increasing_sample_2
# no_signal_1 = [
#     'bin_mle_30_3_False_50_200',
#     'bin_mle_30_3_True_50_200']
# estimators += no_signal_1

estimators = list(set(estimators))
print('Using {0} estimators'.format(len(estimators)))

# %%
id_source = '6'
id_dest = '6'
# load data
ms = []
for i in tqdm(np.arange(len(estimators)), dynamic_ncols=True):
    name = estimators[i]
    ms.append(load(id=id_source, name=name, data=data))

print('Models loaded')


# %%
# selects models in the given order
def select(names, ms):
    ms_ = []
    for name in names:
        run = True
        k = 0
        while (run):
            if ms[k].name == name:
                ms_.append(ms[k])
                run = False
            k += 1
    return ms_


# increasing_parameters = select(ms=ms, names=increasing_parameters_1)
# increasing_parameters = [select(ms=ms, names=increasing_parameters_bin_mle),
#                          select(ms=ms, names=increasing_parameters_bin_mom)]


# %%
#
# increasing_contamination = select(ms=ms,
#                                   names=increasing_contamination_1)  # [select(ms=ms, names=increasing_contamination_1)]
#
# increasing_sample = select(ms=ms, names=increasing_sample_1)  # [select(ms=ms, names=increasing_sample_1),
# # select(ms=ms, names=increasing_sample_2)]
#
# no_signal = select(ms=ms, names=no_signal_1)  # [select(ms=ms, names=no_signal_1)]


# %%


def _save(fig, path):
    fig.savefig(fname=path, bbox_inches='tight')
    plt.close(fig)


def save(fig, name, path):
    Path('./experiments/summaries/{0}'.format(path)).mkdir(parents=True, exist_ok=True)
    path_ = './experiments/summaries/{0}/{1}.pdf'
    _save(fig, path=path_.format(path, name))


def saves(path, names, plots, prefix, hspace, wspace):
    if prefix is not None:
        path_ = './experiments/summaries/{0}/{1}_{2}.pdf'
    else:
        path_ = './experiments/summaries/{0}/{2}.pdf'
    for i, name in enumerate(names):
        fig = plots[i][0]
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        _save(fig, path=path_.format(path, prefix, name))


# %%


def _plot_mean_plus_quantile(ax, x, y, color, label, alpha):
    ax.plot(x, np.mean(y, axis=0), color=color, alpha=1, label=label)
    ax.plot(x, np.quantile(y, q=1 - alpha / 2, axis=0), color=color, alpha=0.2, linestyle='dashed')
    ax.plot(x, np.quantile(y, q=alpha / 2, axis=0), color=color, alpha=0.2, linestyle='dashed')


def __plot_background(func, ax, ms, colors, labels, alpha=0.05, star=True):
    if star:
        ax.hist(ms[0].background.from_,
                ms[0].background.to_,
                weights=ms[0].background.star,
                density=False, color='magenta', alpha=0.05)

    for i in np.arange(len(ms)):
        m = ms[i]
        _plot_mean_plus_quantile(ax=ax,
                                 x=m.background.x_axis,
                                 y=func(m),
                                 color=colors[i],
                                 label=labels[i],
                                 alpha=alpha)

        ax.axvline(x=m.tlower, color='magenta', linestyle='--')
        ax.axvline(x=m.tupper, color='magenta', linestyle='--')

    ax.legend()


def __plot_signal(func, ax, ms, colors, labels, alpha=0.05, star=True):
    if star:
        ax.plot(ms[0].signal.x_axis, ms[0].signal.star, color='magenta')

    for i in np.arange(len(ms)):
        m = ms[i]
        _plot_mean_plus_quantile(ax=ax,
                                 x=m.signal.x_axis,
                                 y=func(m),
                                 color=colors[i], label=labels[i], alpha=alpha)

        # plot signal region
        ax.axvline(x=m.lower, color='magenta', linestyle='--')
        ax.axvline(x=m.upper, color='magenta', linestyle='--')

    ax.legend()


def plot_background(ms, path, colors, labels, alpha=0.05):
    _plot_background = partial(__plot_background, func=lambda m: m.background.predictions)
    _plot_background_error = partial(__plot_background, func=lambda m: m.background.residuals, star=False)
    _plot_signal = partial(__plot_signal, func=lambda m: m.signal.predictions)
    _plot_signal_error = partial(__plot_signal, func=lambda m: m.signal.residuals, star=False)

    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(15, 5))
    _plot_background(ax=axs[0], ms=ms, labels=labels, colors=colors, alpha=alpha)
    _plot_background_error(ax=axs[1], ms=ms, labels=labels, colors=colors, alpha=alpha)
    save(fig=fig, path=path, name='background')

    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(15, 5))
    _plot_signal(ax=axs[0], ms=ms, labels=labels, colors=colors, alpha=alpha)
    _plot_signal_error(ax=axs[1], ms=ms, labels=labels, colors=colors, alpha=alpha)
    save(fig=fig, path=path, name='signal')


def concat(df, dic_):
    return pd.concat([df, pd.DataFrame(dic_, index=[0])],
                     ignore_index=True)


def append(df, method, labels, values, type):
    d = dict(zip(labels, values))
    d['type'] = type
    d['method'] = method
    df_ = concat(df, d)
    return df_


def float_format(x, digits):
    return '%.{0}f'.format(digits) % x


def align_right(str):
    return str.replace('<td>', '<td align="right">')


def html(id, df, name, digits, col_space=100):
    fname = './experiments/summaries/{0}/{1}.html'.format(id, name)
    content = align_right(df.to_html(None,
                                     justify='right',
                                     col_space=col_space,
                                     float_format=lambda x: float_format(x,
                                                                         digits)))

    file = open(fname, 'w')
    file.write(content)
    file.close()


def create_figs(names, n_rows, n_cols, width, height):
    plots = []
    for name in names:
        fig, ax = plt.subplots(n_rows, n_cols,
                               sharey='all',
                               sharex='all',
                               figsize=(width, height))
        fig.suptitle(name)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plots.append((fig, ax.reshape(-1)))
    return plots


def plot_series_with_uncertainty(ax, x, mean, lower, upper, title):
    ax.set_title(title)
    ax.set_xticks(x)
    ax.errorbar(x=x, y=mean,
                yerr=np.vstack((mean - lower, upper - mean)).reshape(2, -1),
                color='black',
                capsize=3)


def create_entry(dotdic, n_rows, n_cols):
    shape = (n_rows, n_cols)
    dotdic.mean = np.zeros(shape, dtype=np.float64)
    dotdic.upper = np.zeros(shape, dtype=np.float64)
    dotdic.lower = np.zeros(shape, dtype=np.float64)


def assing_entry(dotdic, index, mean, lower, upper):
    dotdic.mean[index, :] = mean
    dotdic.lower[index, :] = lower
    dotdic.upper[index, :] = upper


def assing_quantiles(dotdic, index, series, alpha):
    mean = np.mean(series, axis=0).reshape(-1)
    lower = np.quantile(series, q=alpha / 2, axis=0).reshape(-1)
    upper = np.quantile(series, q=1 - alpha / 2, axis=0).reshape(-1)

    assing_entry(dotdic=dotdic, index=index, mean=mean, lower=lower, upper=upper)


def process_data(ms, labels, path, alpha=0.05):
    plt.close('all')  # close all previous figures

    l2_entries = ['l2_back', 'l2_signal', 'chi2_back']

    # l2 = pd.DataFrame(columns=l2_entries + ['method', 'type'])
    bias_table = pd.DataFrame(columns=ESTIMATORS + ['method', 'type'])
    cov_table = pd.DataFrame(columns=CIS_DELTA + ['method', 'type'])

    n_rows = 1
    n_cols = len(ms)
    hspace = 0.4
    wspace = 0.2
    width = 20
    height = 5
    bins = 35
    bias_plots = create_figs(ESTIMATORS, n_rows, n_cols, width, height)
    prob_plots = create_figs(CIS_DELTA, n_rows, n_cols, width, height)
    cov_plots = create_figs(CIS_DELTA, n_rows, n_cols, width, height)
    error_plots = create_figs(ERRORS, n_rows, n_cols, width, height)

    data = DotDic()
    create_entry(data.bias, n_rows=len(ms), n_cols=len(ESTIMATORS))
    create_entry(data.cov, n_rows=len(ms), n_cols=len(CIS_DELTA))
    create_entry(data.width, n_rows=len(ms), n_cols=len(CIS_DELTA))

    for i in np.arange(len(ms)):
        m = ms[i]
        label = labels[i]

        # for plots and table creation

        # normalized bias
        assing_quantiles(data.bias,
                         series=m.bias / m.parameters_star.reshape(1, 4),
                         index=i,
                         alpha=alpha)
        bias_table = append(bias_table, label, ESTIMATORS, data.bias.lower[i, :], 'lower')
        bias_table = append(bias_table, label, ESTIMATORS, data.bias.upper[i, :], 'upper')
        bias_table = append(bias_table, label, ESTIMATORS, data.bias.mean[i, :], 'mean')

        # coverage
        cp = clopper_pearson(m.cov, alpha=alpha)
        assing_entry(data.cov, index=i,
                     mean=np.mean(m.cov, axis=0).reshape(-1),
                     lower=cp[:, 0], upper=cp[:, 1])
        cov_table = append(cov_table, label, CIS_DELTA, data.cov.lower[i, :], 'lower')
        cov_table = append(cov_table, label, CIS_DELTA, data.cov.upper[i, :], 'upper')
        cov_table = append(cov_table, label, CIS_DELTA, data.cov.mean[i, :], 'mean')

        # ci width
        assing_quantiles(data.width,
                         series=m.width.reshape(-1, 4) / m.parameters_star.reshape(1, 4),
                         index=i,
                         alpha=alpha)

        # for dataframe/table creation
        # l2 data
        # l2 = append(l2, name, l2_entries, [np.std(m.l2_background, axis=0), np.std(m.l2_signal, axis=0), 0],
        #                 'std')
        #     l2 = append(l2, name, l2_entries, [np.mean(m.l2_background, axis=0), np.mean(m.l2_signal, axis=0),
        #                                        np.mean(m.chi2_background, axis=0)], 'mean')
        #
        #     # bias data

        # estimate/bias plot
        for j in np.arange(len(PARAMETERS)):
            est = m.est[:, j]

            true_parameter = m[PARAMETERS[j]]
            std_around_true_parameter = np.sqrt(np.sum(np.square(est - true_parameter)) / (est.shape[0] - 1))

            ax = bias_plots[j][1][i]
            ax.set_title(label)
            # if bias_limits[j] is not None:
            #     ax.set_xlim(bias_limits[j])
            # plot.title(ax,
            #            '{0}\nbias {1:.3f} std {2:.3f}'.format(
            #                m.name, bias_.mean[j], bias_.std[j]))

            # plot histogram
            ax.hist(est, bins=bins, density=True)

            # plot pdf around true parameter
            xmin = np.min(est)
            xmax = np.max(est)
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, true_parameter, std_around_true_parameter)
            ax.plot(x, p, 'k', linewidth=2)
            ax.axvline(x=true_parameter, color='red', linestyle='--')

        # prob plots
        for j in np.arange(len(PARAMETERS)):
            est = m.est[:, j]
            ax = prob_plots[j][1][i]
            probplot(x=est, dist="norm", plot=ax)
            ax.set_title(label)
            # plot.title(ax, '{0}\nbias {1:.3f} std {2:.3f}'.format(
            #     m.name, bias_.mean[j], bias_.std[j]))

        # coverage plots
        for j in np.arange(len(PARAMETERS)):
            ax = cov_plots[j][1][i]
            ax.set_title(label)
            # plot.title(ax,
            #            '{0}\nmean {1:.2f} std {2:.2f} width {3:.2f}'.format(
            #                m.name, cov_.mean[j], cov_.std[j], cov_.width[j]))
            for k in np.arange(m.cis.shape[0]):
                ax.plot(m.cis[k, j, :], np.repeat(k, 2) * 0.1,
                        color='blue',
                        alpha=0.5)
                ax.get_yaxis().set_visible(False)  # remove y_axis
            ax.axvline(x=m[PARAMETERS[j]], color='red', linestyle='--')

        # error plots
        for j in np.arange(len(ERRORS)):
            ax = error_plots[j][1][i]
            ax.set_title(label)
            series = m[ERRORS[j]]
            ax.plot(np.arange(series.shape[0]) + 1, series, color='blue')

    saves(path, ESTIMATORS, prob_plots, 'prob', hspace, wspace)
    saves(path, ESTIMATORS, bias_plots, 'bias', hspace, wspace)
    saves(path, CIS_DELTA, cov_plots, 'cov', hspace, wspace)
    saves(path, ERRORS, error_plots, None, hspace, wspace)

    # create hierarchical structure
    bias_table = create_hierarchy(bias_table)
    # l2 = create_hierarchy(l2)
    cov_table = create_hierarchy(cov_table)

    # save statistics
    html(path, bias_table, 'bias', digits=6)
    html(path, cov_table, 'cov', digits=2, col_space=60)
    # html(id_dest, l2, 'l2', digits=5, col_space=60)

    # print("Finish processing data")

    # bias plots
    # fig, axs = plt.subplots(len(ms), len(bias_entries), sharex='col', sharey='col', figsize=(20, 15))
    # bias plots

    # bias, coverage, width plot for delta method confidence interval
    fig, axs = plt.subplots(3, len(ESTIMATORS),
                            sharex='col', sharey='row', figsize=(20, 15))
    for i in np.arange(len(ESTIMATORS)):
        bias_entry = ESTIMATORS[i]
        axs[0, i].axhline(y=0, color='red', linestyle='--')
        plot_series_with_uncertainty(
            ax=axs[0, i],
            title=bias_entry,
            x=labels,
            mean=data.bias.mean[:, i],
            lower=data.bias.lower[:, i],
            upper=data.bias.upper[:, i])

    for i in np.arange(len(CIS_DELTA)):
        cov_entry = CIS_DELTA[i]
        axs[1, i].axhline(y=0.95, color='red', linestyle='--')
        plot_series_with_uncertainty(
            ax=axs[1, i],
            title=cov_entry,
            x=labels,
            mean=data.cov.mean[:, i],
            lower=data.cov.lower[:, i],
            upper=data.cov.upper[:, i])

    for i in np.arange(len(CIS_DELTA)):
        cov_entry = CIS_DELTA[i]
        plot_series_with_uncertainty(
            ax=axs[2, i],
            title=cov_entry,
            x=labels,
            mean=data.width.mean[:, i],
            lower=data.width.lower[:, i],
            upper=data.width.upper[:, i])

    save(fig=fig, path=path, name='bias-cov')
    print("Finish plotting bias and cov")

    # fig, axs = plt.subplots(1, len(l2_entries), sharex='all', figsize=(20, 5))
    # for i in np.arange(len(l2_entries)):
    #     l2_entry = l2_entries[i]
    #     plot_series_with_uncertainty(
    #         ax=axs[i],
    #         title=l2_entry,
    #         x=x_axis,
    #         mean=l2[(l2_entry, 'mean')].values,
    #         uncertainty=l2[(l2_entry, 'std')].values)
    # save_fig(fig=fig, id_dest=id_dest, name='l2')
    # print("Finish plotting l2")
    # plt.close('all')  # close all previous figures


def _plots(ms, path, colors, labels):
    plot_background(ms=ms, path=path, colors=colors, labels=labels)
    process_data(ms=ms, path=path, labels=labels)


_plots(ms=select(ms=ms, names=increasing_parameters_bin_mle),
       path='{0}/parameters/bin_mle'.format(id_dest),
       colors=['red', 'brown', 'orange', 'blue', 'green', 'pink', 'yellow'],
       labels=[4, 5, 10, 20, 30])

# plot_background(ms=select(ms=ms, names=increasing_parameters_bin_mom),
#                 id_dest='5/parameters/bin_mom', from_=from_, to_=to_,
#                 props=props_back, colors=['red', 'orange', 'blue', 'green'],
#                 labels=[5, 10, 30, 40])
# plot_background(ms=select(ms=ms, names=[increasing_parameters_bin_mle[-1], increasing_parameters_bin_mom[-1]]),
#                 id_dest='5/parameters/compare', from_=from_, to_=to_,
#                 props=props_back, colors=['red', 'blue'],
#                 labels=['MLE', 'MOM'])
#


# %%


# process_data(ms=select(ms=ms, names=increasing_parameters_bin_mom),
#              id_dest='5/parameters/bin_mom',
#              x_axis=[5, 10, 30, 40])
# process_data(ms=increasing_contamination, id_dest='4/contamination', x_axis=[1, 5, 32])
# process_data(ms=increasing_sample, id_dest='4/sample', x_axis=[5000, 25000])
# process_data(ms=no_signal, id_dest='4/no_signal', x_axis=['MLE', 'MOM'])
