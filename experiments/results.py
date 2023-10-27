######################################################################
# parse arguments
######################################################################

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--cwd', type=str, default='.')
parser.add_argument('--data_id', type=str, default='3b')
parser.add_argument('--id_dest', type=str, default='test')
parser.add_argument('--a_star', type=int, default=201)
parser.add_argument('--b_star',
                    type=lambda x: None if x == 'None' else int(x),
                    default=None)
parser.add_argument('--rate_star', type=float, default=0.003)
parser.add_argument('--n_bins_star', type=int, default=100)
args, _ = parser.parse_known_args()
print(args)

######################################################################
# Configure matplolib
######################################################################
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, \
    StrMethodFormatter, \
    NullFormatter, \
    FuncFormatter

plt.rcParams['figure.dpi'] = 600
import matplotlib as mpl

#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np, random
import numpy as onp
######################################################################
import math
import pandas as pd
from scipy.stats import probplot
from pathlib import Path
from tqdm import tqdm
from functools import partial

import localize
from src.load import load as load_data
from src.stat.binom import clopper_pearson
from src.dotdic import DotDic
import src.basis.bernstein as basis
from src.transform import transform
from src.bin import proportions, uniform_bin


######################################################################

# %%
# Define estimators of interest


def load(id, name, data):
    if id is not None:
        path = '{0}/experiments/results/{1}/{2}.npz'.format(args.cwd, id, name)
    else:
        path = '{0}/experiments/results/{1}.npz'.format(args.cwd, name)

    file = onp.load(path)
    m = DotDic()
    m.name = name

    m.std_signal_region = int(file['std_signal_region'])
    m.k = int(file['k'])
    m.bins = int(file['bins'])

    m.bias = file['bias']
    m.est = file['estimates']

    m.cov = file['coverage']
    m.width = file['width']
    m.cis = file['cis']

    m.lower = np.float64(file['lower'])
    m.tlower = data.trans(m.lower)
    m.upper = np.float64(file['upper'])
    m.tupper = data.trans(m.upper)

    m.mu_star = np.float64(file['mu_star'])
    m.sigma_star = np.float64(file['sigma_star'])
    m.sigma2_star = np.float64(file['sigma2_star'])
    m.lambda_star = np.float64(file['lambda_star'])
    m.parameters_star = np.array(
        [m.lambda_star, m.lambda_star, m.mu_star, m.sigma2_star],
        dtype=np.float64)

    m.gamma_aux = file['gamma_aux']
    m.gamma_error = m.gamma_aux[:, 0]
    m.gamma_fit = m.gamma_aux[:, 1]
    m.gamma = file['gamma']
    if m.k == 0:
        m.gamma = np.zeros_like(m.gamma_error)

    m.signal_aux = file['signal_aux']
    m.signal_error = m.signal_aux[:, 0]
    m.signal_fit = m.signal_aux[:, 1]

    # data splitting
    idx_sideband = np.array(
        (data.background.X <= m.lower) + (data.background.X >= m.upper),
        dtype=np.bool_)
    idx_signal_region = np.logical_not(idx_sideband)
    data.background.sideband.X = data.background.X[idx_sideband].reshape(-1)
    data.background.signal_region.X = data.background.X[
        idx_signal_region].reshape(-1)
    data.background.trans.sideband.X = data.trans(data.background.sideband.X)
    data.background.trans.signal_region.X = data.trans(
        data.background.signal_region.X)

    # signal density
    n_pos = 100
    m.signal.x_axis = np.linspace(m.mu_star - 3.5 * m.sigma_star,
                                  m.mu_star + 3.5 * m.sigma_star, n_pos)
    m.signal.star = m.lambda_star * normal(m.signal.x_axis, mu=m.mu_star,
                                           sigma2=m.sigma2_star)

    m.signal.predictions = onp.zeros((m.est.shape[0], n_pos))
    m.signal.residuals = onp.zeros((m.est.shape[0], n_pos))
    m.signal.scaled.residuals = onp.zeros((m.est.shape[0], n_pos))

    # background
    m.background.x_axis = data.background.x_axis.reshape(-1)
    m.background.trans.x_axis = data.background.trans.x_axis.reshape(-1)
    m.background.star = data.background.empirical_probabilities.reshape(-1)
    m.background.from_ = data.background.from_.reshape(-1)
    m.background.to_ = data.background.to_.reshape(-1)
    m.background.trans.from_ = data.background.trans.from_.reshape(-1)
    m.background.trans.to_ = data.background.trans.to_.reshape(-1)

    # m.background.trans.chi2.pearson = onp.zeros((m.gamma.shape[0],))
    # m.background.trans.chi2.neyman = onp.zeros((m.gamma.shape[0],))

    m.background.trans.scaled.residuals = onp.zeros(
        (m.gamma.shape[0], m.background.star.shape[0]))
    m.background.trans.residuals = onp.zeros(
        (m.gamma.shape[0], m.background.star.shape[0]))
    m.background.trans.predictions = onp.zeros(
        (m.gamma.shape[0], m.background.star.shape[0]))

    basis_ = basis.integrate(k=m.k,
                             a=data.background.trans.from_,
                             b=data.background.trans.to_)

    # l2 error for background
    # m.background.l2.all = onp.zeros((m.gamma.shape[0],))
    # m.background.l2.sideband = onp.zeros((m.gamma.shape[0],))
    # m.background.l2.signal_region = onp.zeros((m.gamma.shape[0],))
    # m.background.trans.l2.all = onp.zeros((m.gamma.shape[0],))
    # m.background.trans.l2.sideband = onp.zeros((m.gamma.shape[0],))
    # m.background.trans.l2.signal_region = onp.zeros((m.gamma.shape[0],))

    # l2 error for signal
    # m.signal.l2 = onp.zeros((m.gamma.shape[0],))

    # l2 error for mixture
    # m.mixture.l2 = onp.zeros((m.gamma.shape[0],))

    for i in np.arange(m.est.shape[0]):
        gamma = m.gamma[i, :]
        lambda_ = m.est[i, 1]
        mu = m.est[i, 2]
        sigma2 = m.est[i, 3]
        sigma = np.sqrt(sigma2)

        # estimated density
        m.signal.predictions[i, :] = lambda_ * data.signal.density(
            X=m.signal.x_axis, mu=mu, sigma2=sigma2)
        m.signal.residuals[i, :] = m.signal.predictions[i, :] - m.signal.star
        m.signal.scaled.residuals[i, :] = m.signal.predictions[i,
                                          :] / m.signal.star - 1

        # bin background predictions on transformed scale
        m.background.trans.predictions[i, :] = (
                basis_ @ gamma.reshape(-1, 1)).reshape(-1)
        m.background.trans.residuals[i, :] = m.background.trans.predictions[i,
                                             :] - m.background.star
        m.background.trans.scaled.residuals[i,
        :] = m.background.trans.predictions[i, :] / m.background.star - 1

        # background chi2 computations
        # squared_residuals = np.square(m.background.trans.residuals[i, :]).reshape(-1)
        # m.background.trans.pearson_chi2 = np.sum(squared_residuals / m.background.trans.predictions[i, :])
        # m.background.trans.neyman_chi2 = np.sum(squared_residuals / m.background.star)

        # l2 error in background in projected scale
        # background_density = partial(data.background.trans.density, gamma=gamma, k=m.k)
        # m.background.trans.l2.all[i] = l2(func=background_density, X=data.background.trans.X)
        # m.background.trans.l2.sideband[i] = l2_sideband(func=background_density, from_=m.tlower, to_=m.tupper,
        #                                                 X=data.background.trans.sideband.X)
        # m.background.trans.l2.signal_region[i] = l2(func=background_density, from_=m.tlower, to_=m.tupper,
        #                                             X=data.background.trans.signal_region.X)

        # l2 error in background in original scaled
        # background_density = partial(data.background.density, gamma=gamma, k=m.k)
        # m.background.l2.all[i] = l2(func=background_density, X=data.background.X)
        # m.background.l2.sideband[i] = l2_sideband(func=background_density, from_=m.lower, to_=m.upper,
        #                                           X=data.background.sideband.X)
        # m.background.l2.signal_region[i] = l2(func=background_density, from_=m.lower, to_=m.upper,
        #                                       X=data.background.signal_region.X)

        # l2 error in signal, i.e. l2 distance between two normal/gaussian densities
        # sigma2_tilde = sigma2 + m.sigma2_star
        # m.signal.l2[i] = 1 / (2 * np.sqrt(np.pi)) \
        #                  * (1 / sigma + 1 / m.sigma_star) \
        #                  - 2 * normal(X=mu, mu=m.mu_star, sigma2=sigma2_tilde)

        # l2 error of mixture background
        # m.mixture.l2[i] = l2(
        #     func=partial(data.density, gamma=gamma, mu=mu, sigma2=sigma2, lambda_=lambda_, k=m.k),
        #     X=data.X)

    return m


# %%

# load background data
data = DotDic()
data.background.X = load_data(
    '{0}/data/{1}/m_muamu.txt'.format(args.cwd, args.data_id))
print("Data loaded")

# %%
data.background.n = data.background.X.shape[0]
trans, tilt_density, inv_trans = transform(
    a=args.a_star,
    b=args.b_star,
    rate=args.rate_star)

# background in transformed scale
data.trans = trans
from_, to_ = uniform_bin(n_bins=args.n_bins_star)
data.background.trans.from_ = from_
data.background.trans.to_ = to_
data.background.trans.x_axis = (from_ + to_) / 2
data.background.trans.X = trans(data.background.X)
data.background.empirical_probabilities, _ = proportions(
    X=data.background.trans.X,
    from_=data.background.trans.from_,
    to_=data.background.trans.to_)

# axis in original scale
data.background.from_ = inv_trans(from_)
data.background.to_ = onp.array(inv_trans(to_).reshape(-1), dtype=np.float64)
# change infinity for maximum value
data.background.to_[-1] = np.max(data.background.X)
data.background.to_ = np.array(data.background.to_)
data.background.x_axis = (data.background.from_ + data.background.to_) / 2

# add fake signal
# sigma_star = 20.33321
# mu_star = 395.8171
# lambda_star = 0
# sigma2_star = np.square(sigma_star)
# key = random.PRNGKey(seed=0)
#
# data.signal.n = np.int32(data.background.n * lambda_star)
# data.signal.X = mu_star + sigma_star * random.normal(key,
#                                                      shape=(data.signal.n,))
# data.X = np.concatenate((data.background.X, data.signal.X))
# data.empirical_probabilities = proportions(X=trans(data.X),
#                                            from_=from_,
#                                            to_=to_)

# densities
# data.background.trans.density = partial(projected_density,
#                                         basis=basis)
# data.background.density = partial(background,
#                                   tilt_density=tilt_density,
#                                   basis=basis)
data.signal.density = normal

# def mixture_density(X, k, gamma, lambda_, mu, sigma2):
#     scaled_background = (1 - lambda_) * data.background.density(X=X,
#                                                                 gamma=gamma,
#                                                                 k=k)
#     scaled_signal = lambda_ * data.signal.density(X=X,
#                                                   mu=mu,
#                                                   sigma2=sigma2)
#     return scaled_background + scaled_signal
#
#
# data.density = mixture_density
print('Data processed')

# %%
id_source = None
# load data
ms = []
for i in tqdm(np.arange(len(estimators)), dynamic_ncols=True):
    name = estimators[i]
    try:
        ms.append(load(id=id_source, name=name, data=data))
    except FileNotFoundError:
        print("\nFile {0} is missing\n".format(name))
        pass

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


def _save(fig, path):
    fig.savefig(fname=path, bbox_inches='tight')
    plt.close(fig)


def save(fig, name, path):
    Path('{0}/experiments/summaries/{1}'.format(args.cwd, path)).mkdir(
        parents=True,
        exist_ok=True)
    path_ = '{0}/experiments/summaries/{1}/{2}.pdf'
    path_ = path_.format(args.cwd, path, name)
    print(path_)
    _save(fig, path=path_)


def saves(path, names, plots, prefix, hspace, wspace):
    if prefix is not None:
        path_ = '{0}/experiments/summaries/{1}/{2}_{3}.pdf'
    else:
        path_ = '{0}/experiments/summaries/{1}/{3}.pdf'
    for i, name in enumerate(names):
        fig = plots[i][0]
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        _save(fig, path=path_.format(args.cwd, path, prefix, name))


def _plot_mean_plus_quantile(ax, x, y, color, label,
                             alpha=0.05,
                             transparency=0.1):
    ax.plot(x, np.mean(y, axis=0), color=color, alpha=1, label=label)
    ax.fill_between(x, np.mean(y, axis=0),
                    np.quantile(y, q=1 - alpha / 2, axis=0),
                    color=color,
                    alpha=transparency)

    # ax.plot(x, np.quantile(y, q=1 - alpha / 2, axis=0), color=color,
    #         alpha=tranparency, linestyle='dashed')
    # ax.plot(x, np.quantile(y, q=alpha / 2, axis=0), color=color,
    #         alpha=tranparency, linestyle='dashed')


def __plot_background(func, ax, ms, colors, labels, alpha=0.05):
    for i in np.arange(len(ms)):
        m = ms[i]
        _plot_mean_plus_quantile(ax=ax,
                                 x=m.background.trans.x_axis,
                                 y=func(m),
                                 color=colors[i],
                                 label=labels[i],
                                 alpha=alpha,
                                 transparency=0.3)

        ax.axvline(x=m.tlower, color='red', linestyle='--')
        ax.axvline(x=m.tupper, color='red', linestyle='--')

    ax.legend()


def __plot_signal(func, ax, ms, colors, labels, alpha=0.05, star=True):
    if star:
        ax.plot(ms[0].signal.x_axis, ms[0].signal.star, color='black',
                label='True signal')

    for i in np.arange(len(ms)):
        m = ms[i]
        _plot_mean_plus_quantile(ax=ax,
                                 x=m.signal.x_axis,
                                 y=func(m),
                                 color=colors[i], label=labels[i],
                                 alpha=alpha,
                                 transparency=0.4)

        # plot signal region
        ax.axvline(x=m.lower, color='red', linestyle='--')
        ax.axvline(x=m.upper, color='red', linestyle='--')

    ax.legend()


def plot_background_and_signal(ms, path, colors, labels, alpha=0.05,
                               fontsize=20):
    _plot_background = partial(__plot_background,
                               func=lambda m: m.background.trans.predictions)
    _plot_background_error = partial(__plot_background, func=lambda
        m: m.background.trans.scaled.residuals)
    _plot_signal = partial(__plot_signal, func=lambda m: m.signal.predictions)
    _plot_signal_error = partial(__plot_signal,
                                 func=lambda m: m.signal.residuals, star=False)

    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(17, 5))
    fig.suptitle('Background estimation', fontsize=fontsize)
    # left panel
    axs[0].set_xlabel('Mass (projected scale)', fontsize=fontsize)
    axs[0].set_ylabel('Normalized counts', fontsize=fontsize)
    axs[0].hist(ms[0].background.trans.from_,
                np.concatenate((np.array([0]), ms[0].background.trans.to_)),
                # ms[0].background.trans.to_,
                weights=ms[0].background.star,
                density=False,
                color='black',
                alpha=0.05)
    # axs[0].errorbar(x=(ms[0].background.trans.from_ + ms[0].background.trans.to_) / 2,
    #                 y=ms[0].background.star,
    #                 yerr=np.sqrt(ms[0].background.star),
    #                 color='magenta',
    #                 capsize=3)
    _plot_background(ax=axs[0], ms=ms, labels=labels, colors=colors,
                     alpha=alpha)

    # right panel
    axs[1].set_xlabel('Mass (projected scale)', fontsize=fontsize)
    axs[1].set_ylabel('Normalized residuals', fontsize=fontsize)
    axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    _plot_background_error(ax=axs[1], ms=ms, labels=labels, colors=colors,
                           alpha=alpha)
    save(fig=fig, path=path, name='background')
    print('Finish plotting background')

    fig, axs = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(18, 5))
    fig.suptitle('Signal estimation', fontsize=fontsize)
    axs[0].set_xlabel('Mass (GeV)', fontsize=fontsize)
    axs[0].set_ylabel('Density', fontsize=fontsize)
    _plot_signal(ax=axs[0], ms=ms, labels=labels, colors=colors, alpha=alpha)
    axs[1].set_xlabel('Mass (GeV)', fontsize=fontsize)
    axs[1].set_ylabel('Residuals', fontsize=fontsize)
    axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    _plot_signal_error(ax=axs[1], ms=ms, labels=labels, colors=colors,
                       alpha=alpha)
    save(fig=fig, path=path, name='signal')
    print('Finish plotting signal')


def create_hierarchy(df):
    df = df.set_index(['method', 'type']).unstack()
    df.columns = pd.MultiIndex.from_product(df.columns.levels)
    return df


def concat(df, dic_):
    return pd.concat([df, pd.DataFrame(dic_, index=[0])],
                     ignore_index=True)


def append(table, method, labels, values, type):
    d = dict(zip(labels, values))
    d['type'] = type
    d['method'] = method
    df_ = concat(table, d)
    return df_


def float_format(x, digits):
    return '%.{0}f'.format(digits) % x


def align_right(str):
    return str.replace('<td>', '<td align="right">')


def html(id, df, name, digits, col_space=100):
    fname = '{0}/experiments/summaries/{1}/{2}.html'.format(args.cwd, id, name)
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
        fig, axs = plt.subplots(n_rows, n_cols, sharey='row', sharex='row',
                                figsize=(width, height))
        fig.suptitle(name)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plots.append((fig, axs))
    return plots


def plot_series_with_uncertainty(ax, x, mean, lower=None, upper=None):
    mean = mean.reshape(-1)
    ax.set_xticks(x)
    if lower is None and upper is None:
        ax.plot(x, mean, color='black')
    else:
        lower = lower.reshape(-1)
        upper = upper.reshape(-1)
        ax.errorbar(x=x, y=mean,
                    yerr=np.vstack((mean - lower, upper - mean)).reshape(2, -1),
                    color='black',
                    capsize=3)


def create_entry(dotdic, n_rows, n_cols):
    shape = (n_rows, n_cols)
    dotdic.mean = onp.zeros(shape, dtype=np.float64)
    dotdic.upper = onp.zeros(shape, dtype=np.float64)
    dotdic.lower = onp.zeros(shape, dtype=np.float64)


def assing_entry(dotdic, index, mean, lower, upper):
    dotdic.mean[index, :] = mean
    dotdic.lower[index, :] = lower
    dotdic.upper[index, :] = upper


def assing_quantiles(dotdic, index, series, alpha):
    mean = np.mean(series, axis=0).reshape(-1)
    lower = np.quantile(series, q=alpha / 2, axis=0).reshape(-1)
    upper = np.quantile(series, q=1 - alpha / 2, axis=0).reshape(-1)

    assing_entry(dotdic=dotdic, index=index, mean=mean, lower=lower,
                 upper=upper)


def assign_table(table, dotdic, method, index, labels):
    table = append(table=table, method=method, labels=labels,
                   values=dotdic.lower[index, :], type='lower')
    table = append(table=table, method=method, labels=labels,
                   values=dotdic.upper[index, :], type='upper')
    table = append(table=table, method=method, labels=labels,
                   values=dotdic.mean[index, :], type='mean')
    return table


def process_data(ms, labels, alpha=0.05, normalize=True):
    bias_table = pd.DataFrame(columns=ESTIMATORS + ['method', 'type'])
    cov_table = pd.DataFrame(columns=CIS_DELTA + ['method', 'type'])
    width_table = pd.DataFrame(columns=CIS_DELTA + ['method', 'type'])

    data = DotDic()
    create_entry(data.bias, n_rows=len(ms), n_cols=len(ESTIMATORS))
    create_entry(data.cov, n_rows=len(ms), n_cols=len(CIS_DELTA))
    create_entry(data.width, n_rows=len(ms), n_cols=len(CIS_DELTA))
    create_entry(data.l2.background.all, n_rows=len(ms), n_cols=1)
    create_entry(data.l2.background.sideband, n_rows=len(ms), n_cols=1)
    create_entry(data.l2.background.signal_region, n_rows=len(ms), n_cols=1)
    create_entry(data.l2.background.trans.all, n_rows=len(ms), n_cols=1)
    create_entry(data.l2.background.trans.sideband, n_rows=len(ms), n_cols=1)
    create_entry(data.l2.background.trans.signal_region, n_rows=len(ms),
                 n_cols=1)
    create_entry(data.l2.signal, n_rows=len(ms), n_cols=1)
    create_entry(data.l2.mixture, n_rows=len(ms), n_cols=1)
    data.normalize = normalize

    for i in np.arange(len(ms)):
        m = ms[i]
        label = labels[i]

        # bias
        assing_quantiles(dotdic=data.bias, series=m.bias, index=i, alpha=alpha)
        bias_table = assign_table(table=bias_table, dotdic=data.bias,
                                  labels=ESTIMATORS, method=label, index=i)
        shape = data.bias.mean[i, :].shape
        if normalize:
            data.bias.mean[i, :] /= m.parameters_star.reshape(shape)
            data.bias.lower[i, :] /= m.parameters_star.reshape(shape)
            data.bias.upper[i, :] /= m.parameters_star.reshape(shape)

        # coverage
        cp = clopper_pearson(m.cov, alpha=alpha)
        assing_entry(data.cov, index=i,
                     mean=np.mean(m.cov, axis=0).reshape(-1),
                     lower=cp[:, 0],
                     upper=cp[:, 1])
        cov_table = assign_table(table=cov_table, dotdic=data.cov,
                                 labels=CIS_DELTA, method=label, index=i)

        # ci width
        assing_quantiles(data.width, series=m.width, index=i, alpha=alpha)
        width_table = assign_table(table=width_table, dotdic=data.width,
                                   labels=CIS_DELTA, method=label, index=i)
        shape = data.width.mean[i, :].shape
        if normalize:
            data.width.mean[i, :] /= m.parameters_star.reshape(shape)
            data.width.lower[i, :] /= m.parameters_star.reshape(shape)
            data.width.upper[i, :] /= m.parameters_star.reshape(shape)

        # l2
        # assing_quantiles(dotdic=data.l2.background.trans.all,
        #                  series=m.background.trans.l2.all,
        #                  index=i, alpha=alpha)
        # assing_quantiles(dotdic=data.l2.background.trans.signal_region,
        #                  series=m.background.trans.l2.signal_region,
        #                  index=i, alpha=alpha)
        # assing_quantiles(dotdic=data.l2.background.trans.sideband,
        #                  series=m.background.trans.l2.sideband,
        #                  index=i, alpha=alpha)
        # assing_quantiles(dotdic=data.l2.background.all,
        #                  series=m.background.l2.all,
        #                  index=i, alpha=alpha)
        # assing_quantiles(dotdic=data.l2.background.signal_region,
        #                  series=m.background.l2.signal_region,
        #                  index=i, alpha=alpha)
        # assing_quantiles(dotdic=data.l2.background.sideband,
        #                  series=m.background.l2.sideband,
        #                  index=i, alpha=alpha)
        # assing_quantiles(dotdic=data.l2.signal, series=m.signal.l2, index=i,
        #                  alpha=alpha)
        # assing_quantiles(dotdic=data.l2.mixture, series=m.mixture.l2, index=i,
        #                  alpha=alpha)

    print('Finish processing data')

    return data


# def tables(bias_table,cov_table,width_table):
# # create hierarchical structure
# # bias_table = create_hierarchy(bias_table)
# # width_table = create_hierarchy(width_table)
# # cov_table = create_hierarchy(cov_table)
#
# # save statistics
# html(path, bias_table, 'bias', digits=5)
# html(path, cov_table, 'cov', digits=2, col_space=60)
# html(path, width_table, 'width', digits=3, col_space=60)
# print('Finished tables')


def estimates(ms, labels, path):
    ERRORS = ['gamma_error', 'gamma_fit', 'signal_error', 'signal_fit']

    plt.close('all')  # close all previous figures

    hspace = 0.4
    wspace = 0.2
    width = 20
    height = 5
    bins = 35

    bias_plots = create_figs(ESTIMATORS, 2, len(ms), width, height * 2)
    cov_plots = create_figs(CIS_DELTA, 1, len(ms), width, height)
    error_plots = create_figs(ERRORS, 1, len(ms), width, height)

    for i in np.arange(len(ms)):
        m = ms[i]
        label = labels[i]

        # estimate plot
        for j in np.arange(len(PARAMETERS)):
            est = m.est[:, j]
            mean_hat = np.mean(est)
            sigma2_hat = np.sum(np.square(est.reshape(-1) - mean_hat)) / (
                    est.shape[0] - 1)

            ax = bias_plots[j][1][0, i]
            ax.set_title(label)

            # plot histogram
            ax.hist(est, bins=bins, density=True)

            # plot pdf around true parameter
            xmin = np.min(est)
            xmax = np.max(est)
            x = np.linspace(xmin, xmax, 100)
            p = normal(X=x, mu=mean_hat, sigma2=sigma2_hat)
            ax.plot(x, p, 'k', linewidth=2)
            true_parameter = m[PARAMETERS[j]]
            ax.axvline(x=true_parameter, color='red', linestyle='--')

            ax = bias_plots[j][1][1, i]
            probplot(x=est, dist="norm", plot=ax)

        # coverage plots
        for j in np.arange(len(PARAMETERS)):
            ax = cov_plots[j][1][i]
            ax.set_title(label)
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
            ax.hist(m[ERRORS[j]], bins=bins, density=True)
            ax.axvline(x=np.mean(m[ERRORS[j]]), color='red', linestyle='--')

    saves(path, ESTIMATORS, bias_plots, 'bias', hspace, wspace)
    print('Finished plotting estimates')

    saves(path, CIS_DELTA, cov_plots, 'cov', hspace, wspace)
    print('Finished plotting confidence intervals')

    saves(path, ERRORS, error_plots, None, hspace, wspace)
    print('Finished plotting errors')


def is_close_to_int(x, *, atol=1e-10):
    return abs(x - np.round(x)) < atol


class LogFormatterMathtext_(LogFormatter):
    """
    Format values for log axis using ``exponent = log_base(value)``.
    """

    def __call__(self, x, pos=None):

        if x == 0:  # Symlog
            return r'$\mathdefault{0}$'

        sign_string = '-' if x < 0 else ''
        x = abs(x)
        b = self._base

        # only label the decades
        fx = math.log(x) / math.log(b)
        is_x_decade = is_close_to_int(fx)
        exponent = round(fx) if is_x_decade else np.floor(fx)
        coeff = round(b ** (fx - exponent))
        if is_x_decade:
            fx = round(fx)

        if self.labelOnlyBase and not is_x_decade:
            return ''
        if self._sublabels is not None and coeff not in self._sublabels:
            return ''

        # use string formatting of the base if it is not an integer
        if b % 1 == 0.0:
            base = '%d' % b
        else:
            base = '%s' % b

        if abs(fx) < mpl.rcParams['axes.formatter.min_exponent']:
            return r'$\mathdefault{%s%g}$' % (sign_string, x)
        elif not is_x_decade:
            return r'$\mathdefault{%s%s^{%.1f}}$' % (sign_string, base, fx)
        else:
            return r'$\mathdefault{%s%s^{%d}}$' % (sign_string, base, fx)


def set_log_scale(active, ax, x_formatter=None):
    if active:
        ax.set_xscale('log')
        if x_formatter is not None:
            ax.xaxis.set_major_formatter(x_formatter)
        else:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: LogFormatterMathtext_()(x)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        plt.minorticks_off()


def bias_cov(data, fig, axs, labels, xlabel, titles, fontsize, x_log_scale,
             x_formatter):
    axs[0, 0].set_ylabel(
        '{0}mpirical bias'.format('Normalized e' if data.normalize else 'E'),
        fontsize=fontsize)
    axs[1, 0].set_ylabel('Empirical coverage', fontsize=fontsize)

    # make lambda estimators share the y axis of empirical bias
    axs[0, 0].get_shared_y_axes().join(axs[0, 0], axs[0, 1])

    for i in np.arange(len(titles)):
        axs[0, i].axhline(y=0, color='red', linestyle='--')
        axs[0, i].set_title(titles[i], fontsize=fontsize)
        set_log_scale(x_log_scale, axs[0, i], x_formatter)
        plot_series_with_uncertainty(
            ax=axs[0, i],
            x=labels,
            mean=data.bias.mean[:, i]
            # lower=data.bias.lower[:, i],
            # upper=data.bias.upper[:, i]
        )

        axs[1, i].axhline(y=0.95, color='red', linestyle='--')
        # if cov_lim is not None:
        #     axs[1, i].set_ylim(cov_lim)
        # else:
        # axs[1, i].set_ylim([0.8, 1])
        set_log_scale(x_log_scale, axs[1, i], x_formatter)
        plot_series_with_uncertainty(
            ax=axs[1, i],
            x=labels,
            mean=data.cov.mean[:, i],
            lower=data.cov.lower[:, i],
            upper=data.cov.upper[:, i])
    if xlabel is not None:
        if x_log_scale:
            xlabel = "{0} (log scale)".format(xlabel)
        fig.supxlabel(xlabel, fontsize=fontsize)
    # if invert_x_axis:
    #     plt.gca().invert_xaxis()


def metrics(data, labels, path,
            xlabel=None,
            fontsize=20, invert_x_axis=False,
            x_log_scale=False,
            # y_log_scale=False,
            x_formatter=None  # ,
            # bias_lim=None,
            # cov_lim=None
            ):
    # bias, coverage, width plot for delta method confidence interval
    fig, axs = plt.subplots(2, len(ESTIMATORS), sharex='all', sharey='none',
                            figsize=(20, 8))
    bias_cov(data=data,
             fig=fig,
             axs=axs,
             labels=labels,
             xlabel=xlabel,
             titles=[r'$\lambda(\hat{F})$',
                     r'$\lambda_{MLE}(\hat{F})$',
                     r'$\mu_{MLE}(\hat{F})$',
                     r'$\sigma^2_{MLE}(\hat{F})$'],
             fontsize=fontsize,
             x_log_scale=x_log_scale,
             x_formatter=x_formatter)
    save(fig=fig, path=path, name='bias-cov')

    fig, axs = plt.subplots(2, 2,
                            squeeze=False,
                            sharex='all', sharey='none',
                            figsize=(10, 8))
    bias_cov(data=data,
             fig=fig,
             axs=axs,
             labels=labels,
             xlabel=xlabel,
             titles=[r'$\lambda(\hat{F})$'],
             fontsize=fontsize,
             x_log_scale=x_log_scale,
             x_formatter=x_formatter)
    save(fig=fig, path=path, name='reduced-bias-cov')

    print('Finished plotting bias-cov')

    # fig, ax = plt.subplots(1, len(ESTIMATORS), sharex='all', sharey='row', figsize=(20, 10))
    # ax[0].set_ylabel('Normalized empirical width')
    # for i in np.arange(len(ESTIMATORS)):
    #     cov_entry = CIS_DELTA[i]
    #     ax[i].set_title(cov_entry)
    #     plot_series_with_uncertainty(
    #         ax=ax,
    #         x=labels,
    #         mean=data.width.mean[:, i],
    #         lower=data.width.lower[:, i],
    #         upper=data.width.upper[:, i])
    # save(fig=fig, path=path, name='width')

    # fig, axs = plt.subplots(3, 3, sharex='none', sharey='none', figsize=(20, 15))
    # axs[0, 0].set_title('projected background (all)')
    # plot_series_with_uncertainty(
    #     ax=axs[0, 0],
    #     title=,
    #     x=labels,
    #     mean=data.l2.background.trans.all.mean,
    #     lower=data.l2.background.trans.all.lower,
    #     upper=data.l2.background.trans.all.upper)
    # plot_series_with_uncertainty(
    #     ax=axs[0, 1],
    #     title='projected background (sideband)',
    #     x=labels,
    #     mean=data.l2.background.trans.sideband.mean,
    #     lower=data.l2.background.trans.sideband.lower,
    #     upper=data.l2.background.trans.sideband.upper)
    # plot_series_with_uncertainty(
    #     ax=axs[0, 2],
    #     title='projected background (signal)',
    #     x=labels,
    #     mean=data.l2.background.trans.signal_region.mean,
    #     lower=data.l2.background.trans.signal_region.lower,
    #     upper=data.l2.background.trans.signal_region.upper)
    # plot_series_with_uncertainty(
    #     ax=axs[1, 0],
    #     title='background (all)',
    #     x=labels,
    #     mean=data.l2.background.all.mean,
    #     lower=data.l2.background.all.lower,
    #     upper=data.l2.background.all.upper)
    # plot_series_with_uncertainty(
    #     ax=axs[1, 1],
    #     title='background (sideband)',
    #     x=labels,
    #     mean=data.l2.background.sideband.mean,
    #     lower=data.l2.background.sideband.lower,
    #     upper=data.l2.background.sideband.upper)
    # plot_series_with_uncertainty(
    #     ax=axs[1, 2],
    #     title='background (signal)',
    #     x=labels,
    #     mean=data.l2.background.signal_region.mean,
    #     lower=data.l2.background.signal_region.lower,
    #     upper=data.l2.background.signal_region.upper)
    # plot_series_with_uncertainty(
    #     ax=axs[2, 0],
    #     title='signal',
    #     x=labels,
    #     mean=data.l2.signal.mean,
    #     lower=data.l2.signal.lower,
    #     upper=data.l2.signal.upper)
    #
    # plot_series_with_uncertainty(
    #     ax=axs[2, 1],
    #     title='mixture',
    #     x=labels,
    #     mean=data.l2.mixture.mean,
    #     lower=data.l2.mixture.lower,
    #     upper=data.l2.mixture.upper)
    #
    # save(fig=fig, path=path, name='l2')
    # print('Finished plotting l2')

    plt.close('all')  # close all previous figures


# %%
# bias-coverage as K increases, fixed n
ms_ = select(ms=ms, names=increasing_parameters)
labels_ = INCREASING_PARAMETERS
data_ = process_data(ms=ms_, labels=labels_, normalize=False)

metrics(data=data_,
        path='{0}/parameters/bin_mle'.format(args.id_dest),
        labels=labels_,
        xlabel='background complexity (K)')

estimates(ms=ms_,
          labels=labels_,
          path='{0}/parameters/bin_mle/'.format(args.id_dest))

# %%

# background and signal estimation as K increases, fixed n
ms_ = select(ms=ms, names=reduced_parameters)
labels_ = REDUCED_PARAMETERS
data_ = process_data(ms=ms_, labels=labels_, normalize=False)

plot_background_and_signal(ms=ms_,
                           path='{0}/parameters/bin_mle/reduced'.format(
                               args.id_dest),
                           colors=['blue',
                                   'red',
                                   'green',
                                   'magenta'],
                           labels=['K={0}'.format(k) for k in
                                   REDUCED_PARAMETERS])

metrics(data=data_,
        path='{0}/parameters/bin_mle/reduced'.format(args.id_dest),
        labels=labels_,
        xlabel='background complexity (K)')

estimates(ms=ms_,
          labels=labels_,
          path='{0}/parameters/bin_mle/reduced'.format(args.id_dest))

# %%
# bias-coverage as n increases, fixed K
ms_ = select(ms=ms, names=increasing_sample)
labels_ = INCREASING_SAMPLE
data_ = process_data(ms=ms_, labels=labels_, normalize=False)
metrics(data=data_,
        path='{0}/sample/bin_mle'.format(args.id_dest),
        labels=labels_,
        xlabel='folds',
        x_log_scale=False)

# %%
# bias-coverage as lambda->0, fixed K
labels_ = DIMINISHING_SIGNAL * 100
ms_ = select(ms=ms, names=diminishing_signal)
data_ = process_data(ms=ms_, labels=labels_, normalize=False)

metrics(data=data_,
        path='{0}/signal/bin_mle'.format(args.id_dest),
        labels=labels_,
        xlabel='Percentage of data from the signal')  # r'$\lambda$',
# x_log_scale=True,
# x_formatter=StrMethodFormatter('{x:.02}'))

estimates(ms=ms_,
          labels=labels_,
          path='{0}/signal/bin_mle'.format(args.id_dest))

# # %%
# bias-coverage as the contamination is increased, fixed K, fixed n
from jax.scipy.stats.norm import cdf

contamination = lambda x: np.round((1 - (cdf(x) - cdf(-x))) * 100, 1)
ms_ = select(ms=ms, names=increasing_contamination)
labels_ = contamination(INCREASING_CONTAMINATION)
data_ = process_data(ms=ms_, labels=labels_, normalize=False)

metrics(data=data_,
        path='{0}/contamination/bin_mle'.format(args.id_dest),
        labels=labels_,
        xlabel='Percentage of signal outside signal region',
        # r'$S_{\theta}(C)\approx\epsilon\%$',
        x_log_scale=True,
        x_formatter=StrMethodFormatter('{x:.0f}'))
