# %%

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 600
import jax.random as random
from scipy.stats import norm, probplot
from src.stat.binom import clopper_pearson, wilson
from src.dotdic import DotDic
from src.predict import predict
import src.basis.bernstein as basis
from src.transform import transform
import src.plot as plot
from pathlib import Path
from src.bin import proportions, uniform_bin
import os
from tqdm import tqdm

os.chdir('./experiments/')  # when working locally


# %%

# TODO: use quantiles instead of symmetric SD
# TODO: use plots instead of tables
# TODO: create plots for the predictions of each model
# TODO: plot metrics of diff model classes (mle vs bin_mom) in the same plot (e.g. cov and bias)


# %%

def concat(df, dic_):
    return pd.concat([df, pd.DataFrame(dic_, index=[0])],
                     ignore_index=True)


def append(df, method, labels, values, type):
    d = dict(zip(labels, values))
    d['type'] = type
    d['method'] = method
    df_ = concat(df, d)
    return df_


def gaussian(x, mu, sigma):
    return np.exp(- np.square(x - mu) / (2 * np.square(sigma))) / (
            np.sqrt(2 * np.pi) * sigma)


def load(id, name, X, props_back):
    file = np.load('./results/{0}/{1}.npz'.format(id, name))
    m = DotDic()
    m.name = name

    m.gamma = file['gamma']

    m.bias = file['bias']
    m.est = file['estimates']

    m.cov = file['coverage']
    m.width = file['width']
    m.cis = file['cis']

    m.signal_region = file['signal_region']
    m.std_signal_region = int(file['std_signal_region'])
    m.k = int(file['k'])
    m.bins = int(file['bins'])

    m.mu_star = np.float64(file['mu_star'])
    m.sigma_star = np.float64(file['sigma_star'])
    m.sigma2_star = np.square(m.sigma_star)
    m.lambda_star = np.float64(file['lambda_star'])

    m.gamma_error = file['gamma_error']
    m.signal_error = file['signal_error']

    # l2 error computation
    M = basis.integrated_inner_product(m.k)

    m.alt_chi2_background = np.zeros(m.gamma.shape[0])
    m.chi2_background = np.zeros(m.gamma.shape[0])
    m.residual_background = np.zeros((m.gamma.shape[0], props_back.shape[0]))
    m.preds_background = np.zeros((m.gamma.shape[0], props_back.shape[0]))
    m.l2_background = np.zeros(m.gamma.shape[0])

    m.l2_signal = np.zeros(m.gamma.shape[0])

    # error and chi2 computation
    props_back = props_back.reshape(-1)
    basis_ = basis.integrate(k=m.k, a=from_, b=to_)

    for i in np.arange(m.gamma.shape[0]):
        gamma = m.gamma[i, :]
        lambda_ = m.est[i, 1]
        mu = m.est[i, 2]
        sigma = m.est[i, 3]

        # TODO: compute l2 error for the mixture
        # TODO: compute the l2 error in the original scale of the data
        # l2 error in background
        avg = np.mean(
            (basis.evaluate(k=m.k, X=X) @ gamma.reshape(-1, 1)).reshape(-1))
        m.l2_background[i] = gamma.reshape((1, -1)) @ M @ gamma.reshape(
            (-1, 1)) - 2 * avg

        # l2 error in signal (gaussian)
        sigma_tilde = np.sqrt(np.square(sigma) + np.square(m.sigma_star))
        m.l2_signal[i] = 1 / (2 * np.sqrt(np.pi)) \
                         * (1 / sigma + 1 / m.sigma_star) \
                         - 2 * gaussian(x=mu, mu=m.mu_star, sigma=sigma_tilde)

        # residuals and chi2 for background
        preds_back = (basis_ @ gamma.reshape(-1, 1)).reshape(-1)
        errors_back = (props_back - preds_back)
        chi2_back = np.sum(np.square(errors_back) / preds_back)
        alt_chi2_back = np.sum(np.square(errors_back) / props_back)

        m.chi2_background[i] = chi2_back
        m.alt_chi2_background[i] = alt_chi2_back
        m.residual_background[i, :] = errors_back
        m.preds_background[i, :] = preds_back

        # residuals and chi2 for background and signal

        # preds_backsig = (1-lambda_) *  (basis_ @ gamma.reshape(-1, 1)).reshape(-1) + lambda_ *
        # errors_backsig = (props_backsig - preds_backsig)
        #
        # m.residual_backsig[i, :] = errors_back
        # m.preds_backsig[i, :] = preds_back

    print('{0} loaded'.format(m.name))
    return m


def create_hierarchy(df):
    df = df.set_index(['method', 'type']).unstack()
    df.columns = pd.MultiIndex.from_product(df.columns.levels)
    return df


# %%

# load background data and fake signal
X = np.array(np.loadtxt('../data/1/m_muamu.txt'))
trans, tilt_density = transform(X)
tX = trans(X)
from_, to_ = uniform_bin(step=0.01)
props_back = proportions(X=tX, from_=from_, to_=to_)

seed = 0
sigma_star = 20
mu_star = 450
lambda_star = 0.01

n = X.shape[0]
n_signal = np.int32(n * lambda_star)
key = random.PRNGKey(seed=seed)
signal = mu_star + sigma_star * random.normal(key, shape=(n_signal,))
X_with_signal = np.concatenate((X, signal))
tX_with_signal = trans(X_with_signal)
props_back_and_sig = proportions(X=tX_with_signal, from_=from_, to_=to_)

print("Data loaded")

# %%
#

colors = {'bin_mle_5_3_False_50_200': 'red',  # increasing parameters
          'bin_mle_10_3_False_50_200': 'yellow',
          'bin_mle_20_3_False_50_200': 'orange',
          'bin_mle_30_3_False_50_200': 'blue',
          'bin_mle_50_3_False_50_200': 'green',

          'bin_mle_30_3_True_50_200': 'red',  # no signal

          'bin_mle_30_2_False_50_200': 'red',  # increasing contamination
          'bin_mle_30_1_False_50_200': 'green',

          'bin_mle_30_3_False_1_200': 'red',  # for sample size experiment
          'bin_mle_30_3_False_5_200': 'green',

          'bin_mom_lawson_scipy_20_3_False_50_200': 'red'  # compare models
          }

# %%

estimators = []
increasing_parameters_1 = [
    'bin_mle_5_3_False_50_200',
    'bin_mle_10_3_False_50_200',
    'bin_mle_20_3_False_50_200',
    'bin_mle_30_3_False_50_200',
    'bin_mle_50_3_False_50_200']
estimators += increasing_parameters_1
increasing_parameters_2 = [
    'bin_mle_5_3_False_50_200',
    'bin_mle_10_3_False_50_200',
    'bin_mle_20_3_False_50_200',
    'bin_mle_30_3_False_50_200']
estimators += increasing_parameters_2
increasing_contamination_1 = [
    'bin_mle_30_3_False_50_200',
    'bin_mle_30_2_False_50_200',
    'bin_mle_30_1_False_50_200']
estimators += increasing_contamination_1
increasing_sample_1 = [
    'bin_mle_30_3_False_5_200',
    'bin_mle_30_3_False_50_200']
estimators += increasing_sample_1
increasing_sample_2 = [
    'bin_mom_lawson_scipy_30_3_False_5_200',
    'bin_mom_lawson_scipy_30_3_False_50_200']
estimators += increasing_sample_2
no_signal_1 = [
    'bin_mle_30_3_False_50_200',
    'bin_mle_30_3_True_50_200']
estimators += no_signal_1

estimators = list(set(estimators))
print('Using {0} estimators'.format(len(estimators)))

# %%
id_source = '2'

# load data
ms = []
for i in tqdm(np.arange(len(estimators)), dynamic_ncols=True):
    name = estimators[i]
    ms.append(load(id=id_source, name=name, X=tX, props_back=props_back))

print('Models loaded')


# %%


def select(names, ms):
    ms_ = []
    for m in ms:
        if m.name in names:
            ms_.append(m)
    return ms_


increasing_parameters = select(ms=ms, names=increasing_parameters_1)
# increasing_parameters = [select(ms=ms, names=increasing_parameters_1),
#                          select(ms=ms, names=increasing_parameters_2)]

increasing_contamination = select(ms=ms,
                                  names=increasing_contamination_1)  # [select(ms=ms, names=increasing_contamination_1)]

increasing_sample = select(ms=ms, names=increasing_sample_1)  # [select(ms=ms, names=increasing_sample_1),
# select(ms=ms, names=increasing_sample_2)]

no_signal = select(ms=ms, names=no_signal_1)  # [select(ms=ms, names=no_signal_1)]


# %%


# %%

def save_fig(fig, name, id_dest):
    Path('./summaries/{0}'.format(id_dest)).mkdir(parents=True, exist_ok=True)
    path = './summaries/{0}/{1}.pdf'
    plot.save(fig, path=path.format(id_dest, name))


# %%


def plot_background(ms, id_dest, from_, to_, props):
    x = (from_ + to_) / 2

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    ax.hist(from_, to_, weights=props, density=False, color='magenta', alpha=0.2)

    for i in np.arange(len(ms)):
        m = ms[i]
        # for j in np.arange(m.residual_background.shape[0]):
        #     ax.plot(x, m.preds_background[j, :], color=colors[m.name], alpha=0.03)
        # plot mean error
        mean = np.mean(m.preds_background, axis=0)
        std = np.std(m.preds_background, axis=0)
        ax.plot(x, mean, color=colors[m.name], alpha=1, label=m.name)
        ax.plot(x, mean + std, color=colors[m.name], alpha=0.3, linestyle='dashed')
        ax.plot(x, mean - std, color=colors[m.name], alpha=0.3, linestyle='dashed')

        # plot signal region
        lower, upper = m.signal_region[0]
        lower = trans(lower)
        upper = trans(upper)
        ax.axvline(x=lower, color='magenta', linestyle='--')
        ax.axvline(x=upper, color='magenta', linestyle='--')

    ax.legend()
    # plt.show(block=True)
    # ax.set_xlim(left=lower, right=upper)
    save_fig(fig=fig, id_dest=id_dest, name='background')


plot_background(ms=increasing_parameters, id_dest='4/parameters', from_=from_, to_=to_, props=props_back)
plot_background(ms=increasing_contamination, id_dest='4/contamination', from_=from_, to_=to_, props=props_back)
plot_background(ms=increasing_sample, id_dest='4/sample', from_=from_, to_=to_, props=props_back)
plot_background(ms=no_signal, id_dest='4/signal', from_=from_, to_=to_, props=props_back)


# %%

def plot_background_error(ms, id_dest, from_, to_):
    x = (from_ + to_) / 2

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for i in np.arange(len(ms)):
        m = ms[i]
        # for j in np.arange(m.residual_background.shape[0]):
        #     ax.plot(x, m.residual_background[j, :], color=m.color, alpha=0.01)
        # plot mean error
        mean = np.mean(m.residual_background, axis=0)
        std = np.std(m.residual_background, axis=0)
        ax.plot(x, mean, color=colors[m.name], label=m.name, alpha=1)
        ax.plot(x, mean + std, color=colors[m.name], alpha=0.2, linestyle='dashed')
        ax.plot(x, mean - std, color=colors[m.name], alpha=0.2, linestyle='dashed')

        # plot signal region
        lower, upper = m.signal_region[0]
        lower = trans(lower)
        upper = trans(upper)
        ax.axvline(x=lower, color='magenta', linestyle='--')
        ax.axvline(x=upper, color='magenta', linestyle='--')

    ax.legend()
    # plt.show(block=True)
    save_fig(fig=fig, id_dest=id_dest, name='background_error')

    # fig = plt.figure(figsize=(7, 5))
    # ax = fig.add_subplot(111)
    #
    # for i in np.arange(len(ms)):
    #     m = ms[i]
    #     ax.hist(m.chi2_background, bins=20, color=m.color, alpha=0.5, label=m.name)
    # ax.legend()
    # plt.show(block=True)


plot_background_error(ms=increasing_parameters, id_dest='4/parameters', from_=from_, to_=to_)
plot_background_error(ms=increasing_contamination, id_dest='4/contamination', from_=from_, to_=to_)
plot_background_error(ms=increasing_sample, id_dest='4/sample', from_=from_, to_=to_)
plot_background_error(ms=no_signal, id_dest='4/signal', from_=from_, to_=to_)


# %%

def plot_signal(ms, id_dest):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    mu_star = ms[0].mu_star
    sigma_star = ms[0].sigma_star
    lambda_star = ms[0].lambda_star
    n_pos = 100
    x = np.linspace(mu_star - 3.5 * sigma_star, mu_star + 3.5 * sigma_star, n_pos)
    ax.plot(x, lambda_star * norm.pdf(x, loc=mu_star, scale=sigma_star), color='magenta')

    for i in np.arange(len(ms)):
        m = ms[i]
        color = colors[m.name]
        densities = np.zeros((m.est.shape[0], n_pos))
        for j in np.arange(m.est.shape[0]):
            densities[j, :] = m.est[j, 1] * norm.pdf(x, loc=m.est[j, 2], scale=np.sqrt(m.est[j, 3]))
            # ax.plot(x, densities[j, :], color=color, alpha=0.01)
        # plot mean
        mean = np.mean(densities, axis=0)
        std = np.std(densities, axis=0)
        ax.plot(x, mean, color=color, alpha=1, label=m.name)
        ax.plot(x, mean + std, color=color, alpha=0.2, linestyle='dashed')
        ax.plot(x, np.maximum(mean - std, 0), color=color, alpha=0.2, linestyle='dashed')
        # plot signal region
        lower, upper = m.signal_region[0]
        ax.axvline(x=lower, color='magenta', linestyle='--')
        ax.axvline(x=upper, color='magenta', linestyle='--')

    ax.legend()
    # plt.show(block=True)
    save_fig(fig=fig, id_dest=id_dest, name='signal')


plot_signal(ms=increasing_parameters, id_dest='4/parameters')
plot_signal(ms=increasing_contamination, id_dest='4/contamination')
plot_signal(ms=increasing_sample, id_dest='4/sample')
plot_signal(ms=no_signal, id_dest='4/signal')


# %%

def float_format(x, digits):
    return '%.{0}f'.format(digits) % x


def align_right(str):
    return str.replace('<td>', '<td align="right">')


def html(id, df, name, digits, col_space=100):
    fname = './summaries/{0}/{1}.html'.format(id, name)
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


def save_figs(id, estimators, plots, prefix, hspace, wspace):
    Path('./summaries/{0}'.format(id)).mkdir(parents=True, exist_ok=True)
    if prefix is not None:
        path = './summaries/{0}/{1}_{2}.pdf'
    else:
        path = './summaries/{0}/{2}.pdf'
    for i, estimator in enumerate(estimators):
        fig = plots[i][0]
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        plot.save(fig, path=path.format(id, prefix, estimator))


def plot_series_with_uncertainty(ax, x, mean, uncertainty, title):
    ax.set_title(title)
    ax.errorbar(x=x, y=mean, yerr=uncertainty, color='black', capsize=3)


def process_data(ms, x_axis, id_dest):
    bias_parameters = ['lambda_star', 'lambda_star', 'mu_star', 'sigma2_star']
    cov_parameters = ['lambda_star', 'lambda_star', 'mu_star', 'sigma2_star']
    errors = ['gamma_error', 'signal_error']

    bias_entries = ['lambda_hat0', 'lambda_hat', 'mu_hat', 'sigma2_hat']
    cov_entries = ['lambda_hat0_delta', 'lambda_hat_delta', 'mu_hat_delta', 'sigma2_hat_delta']
    l2_entries = ['l2_back', 'l2_signal', 'chi2_back']

    l2 = pd.DataFrame(columns=l2_entries + ['method', 'type'])
    bias = pd.DataFrame(columns=bias_entries + ['method', 'type'])
    cov = pd.DataFrame(columns=cov_entries + ['method', 'type'])

    n_rows = 1
    n_cols = len(ms)
    hspace = 0.4
    wspace = 0.2
    width = 20
    height = 5
    bins = 35
    bias_plots = create_figs(bias_entries, n_rows, n_cols, width, height)
    prob_plots = create_figs(cov_entries, n_rows, n_cols, width, height)
    cov_plots = create_figs(cov_entries, n_rows, n_cols, width, height)
    error_plots = create_figs(errors, n_rows, n_cols, width, height)

    for i in np.arange(len(ms)):
        m = ms[i]
        name = m.name

        # l2 data
        l2 = append(l2, name, l2_entries, [np.std(m.l2_background, axis=0), np.std(m.l2_signal, axis=0), 0],
                    'std')
        l2 = append(l2, name, l2_entries, [np.mean(m.l2_background, axis=0), np.mean(m.l2_signal, axis=0),
                                           np.mean(m.chi2_background, axis=0)], 'mean')

        # bias data
        bias = append(bias, name, bias_entries, np.std(m.bias, axis=0), 'std')
        bias = append(bias, name, bias_entries, np.mean(m.bias, axis=0), 'mean')

        # coverage data
        cov = append(cov, name, cov_entries, np.mean(m.width, axis=0), 'width')
        cov = append(cov, name, cov_entries, np.std(m.cov, axis=0), 'std')
        cov = append(cov, name, cov_entries, np.mean(m.cov, axis=0), 'mean')

        cp = clopper_pearson(m.cov)
        cov = append(cov, name, cov_entries, cp[:, 1], 'CP_up')
        cov = append(cov, name, cov_entries, cp[:, 0], 'CP_low')

        # wil = wilson(m.cov)
        # cov = append(cov, m.name, cov_entries, wil[:, 1], 'Wil_up')
        # cov = append(cov, m.name, cov_entries, wil[:, 0], 'Wil_low')

        # bias plots
        for j in np.arange(len(bias_parameters)):
            est = m.est[:, j]

            true_parameter = m[bias_parameters[j]]
            std_around_true_parameter = np.sqrt(
                np.sum(np.square(est - true_parameter)) / (est.shape[0] - 1))

            ax = bias_plots[j][1][i]
            ax.set_title(name)
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
        for j in np.arange(len(bias_parameters)):
            est = m.est[:, j]
            ax = prob_plots[j][1][i]
            probplot(x=est, dist="norm", plot=ax)
            ax.set_title(name)
            # plot.title(ax, '{0}\nbias {1:.3f} std {2:.3f}'.format(
            #     m.name, bias_.mean[j], bias_.std[j]))

        # coverage plots
        for j in np.arange(len(cov_parameters)):
            ax = cov_plots[j][1][i]
            ax.set_title(name)
            # plot.title(ax,
            #            '{0}\nmean {1:.2f} std {2:.2f} width {3:.2f}'.format(
            #                m.name, cov_.mean[j], cov_.std[j], cov_.width[j]))
            for k in np.arange(m.cis.shape[0]):
                ax.plot(m.cis[k, j, :], np.repeat(k, 2) * 0.1,
                        color='blue',
                        alpha=0.5)
                plot.remove_y_axis(ax)
            ax.axvline(x=m[cov_parameters[j]], color='red', linestyle='--')

        # error plots
        for j in np.arange(len(errors)):
            ax = error_plots[j][1][i]
            ax.set_title(name)
            plot.title(ax, m.name)
            series = m[errors[j]]
            ax.plot(np.arange(series.shape[0]) + 1, series, color='blue')

    save_figs(id_dest, bias_entries, prob_plots, 'prob', hspace, wspace)
    save_figs(id_dest, bias_entries, bias_plots, 'bias', hspace, wspace)
    save_figs(id_dest, cov_entries, cov_plots, 'cov', hspace, wspace)
    save_figs(id_dest, errors, error_plots, None, hspace, wspace)
    plt.close('all')  # close all previous figures

    # create hierarchical structure
    bias = create_hierarchy(bias)
    l2 = create_hierarchy(l2)
    cov = create_hierarchy(cov)

    # save statistics
    html(id_dest, bias, 'bias', digits=6)
    html(id_dest, cov, 'cov', digits=2, col_space=60)
    html(id_dest, l2, 'l2', digits=5, col_space=60)

    print("Finish processing data")

    fig, axs = plt.subplots(1, 4, sharex='all', figsize=(20, 5))
    for i in np.arange(len(bias_entries)):
        bias_entry = bias_entries[i]
        axs[i].axhline(y=0, color='red', linestyle='--')
        plot_series_with_uncertainty(
            ax=axs[i],
            title=bias_entry,
            x=x_axis,
            mean=bias[(bias_entry, 'mean')].values,
            uncertainty=bias[(bias_entry, 'std')].values)
    save_fig(fig=fig, id_dest=id_dest, name='bias')

    print("Finish plotting bias")

    fig, axs = plt.subplots(1, 4, sharex='all', sharey='all', figsize=(20, 5))
    for i in np.arange(len(cov_entries)):
        cov_entry = cov_entries[i]
        mean = cov[(cov_entry, 'mean')].values
        axs[i].axhline(y=0.95, color='red', linestyle='--')
        plot_series_with_uncertainty(
            ax=axs[i],
            title=cov_entry,
            x=x_axis,
            mean=mean,
            uncertainty=np.vstack((mean - cov[(cov_entry, 'CP_low')].values,
                                   cov[(cov_entry, 'CP_up')].values - mean)))
    save_fig(fig=fig, id_dest=id_dest, name='cov')
    print("Finish plotting covariance")

    fig, axs = plt.subplots(1, 2, sharex='all', figsize=(20, 5))
    for i in np.arange(len(l2_entries)):
        l2_entry = l2_entries[i]
        plot_series_with_uncertainty(
            ax=axs[i],
            title=l2_entry,
            x=x_axis,
            mean=l2[(l2_entry, 'mean')].values,
            uncertainty=l2[(l2_entry, 'std')].values)
    save_fig(fig=fig, id_dest=id_dest, name='l2')
    print("Finish plotting l2")
    plt.close('all')  # close all previous figures


process_data(ms=increasing_parameters, id_dest='4/parameters', x_axis=[5, 10, 20, 30, 50])
process_data(ms=increasing_contamination, id_dest='4/contamination', x_axis=[1, 5, 32])
process_data(ms=increasing_sample, id_dest='4/sample', x_axis=[5000, 25000])
process_data(ms=no_signal, id_dest='4/no_signal', x_axis=['MLE', 'MOM'])
