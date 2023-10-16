from src.load import load
from src.random import add_signal
import jax.random as random
import jax.numpy as np
from tqdm import tqdm
from src.dotdic import DotDic
from src.ci.delta import delta_cis
import numpy as onp


def _evaluate(X, params):
    method = DotDic()
    method.name = params.name
    method.k = params.k

    add_signal(X=X, params=params, method=method)

    if params.model_selection:
        val_error = onp.zeros(len(params.k_grid))
        for i, k in enumerate(params.k_grid):
            params.k = k
            params.background.fit(params=params, method=method)
            val_error[i] = method.background.validation_error()
        k_star = onp.argmin(val_error)
        params.k = params.k_grid[k_star]

    params.background.fit(params=params, method=method)

    params.signal.fit(params=params, method=method)

    delta_cis(params=params, method=method)

    return method


def evaluate(X, params):
    method = _evaluate(X, params=params)

    #######################################################
    # save results
    #######################################################
    save = DotDic()

    save.bias = onp.zeros((len(params.estimators),), dtype=params.dtype)
    save.estimates = onp.zeros((len(params.estimators),), dtype=params.dtype)
    save.coverage = onp.zeros((len(params.cis),), dtype=onp.bool_)
    save.width = onp.zeros((len(params.cis),), dtype=params.dtype)
    save.cis = onp.zeros((len(params.cis), 2), dtype=params.dtype)

    save.gamma = method.gamma
    save.gamma_aux = method.gamma_aux
    save.signal_aux = method.signal_aux

    # bias
    for i in np.arange(len(params.estimators)):
        estimated_parameter = method[params.estimators[i]]
        save.bias[i] = estimated_parameter - params.true_parameters[i]
        save.estimates[i] = estimated_parameter

    # coverage
    for i in np.arange(len(params.cis)):
        parameter = params.ci_parameters[i]
        ci = method[params.cis[i]]
        save.coverage[i] = (ci[0] <= parameter) and (parameter <= ci[1])
        save.width[i] = ci[1] - ci[0]
        save.cis[i, :] = ci

    # plots related to one of the iterations

    return save


def run(params):
    #######################################################
    # load background data
    #######################################################
    X_ = load(params.data)

    #######################################################
    # split data
    #######################################################
    print("Original data size: {0}".format(X_.shape[0]))
    print("Folds: {0}".format(params.folds))
    idx = random.permutation(key=params.key,
                             x=np.arange(X_.shape[0]),
                             independent=True)
    idxs = np.array_split(idx, indices_or_sections=params.folds)

    #######################################################
    # print info
    #######################################################
    print('\n{0}'.format(params.name))
    print('Data source: {0}'.format(params.data))
    print('Using {0} datasets'.format(params.folds))
    print('Datasets have {0} examples'.format(X_[idxs[0]].shape[0]))
    print('Model selection: {0}'.format(params.model_selection))

    if params.sample_split:
        print('Using sample splitting')

    save = DotDic()
    save.bias = onp.zeros((params.folds, len(params.estimators)),
                          dtype=params.dtype)
    save.coverage = onp.zeros((params.folds, len(params.cis)), dtype=onp.bool_)
    save.width = onp.zeros((params.folds, len(params.cis)), dtype=params.dtype)
    save.estimates = onp.zeros((params.folds, len(params.estimators)),
                               dtype=params.dtype)
    save.cis = onp.zeros((params.folds, len(params.cis), 2), dtype=params.dtype)
    save.gamma = onp.zeros((params.folds, params.k + 1), dtype=params.dtype)
    save.gamma_aux = onp.zeros((params.folds, 2), dtype=params.dtype)
    save.signal_aux = onp.zeros((params.folds, 2), dtype=params.dtype)

    for i in tqdm(range(params.folds),
                  dynamic_ncols=True):
        save_ = evaluate(X=X_[idxs[i]], params=params)
        save.bias[i, :] = save_.bias
        save.coverage[i, :] = save_.coverage
        save.width[i, :] = save_.width
        save.estimates[i, :] = save_.estimates
        save.cis[i, :, :] = save_.cis
        save.gamma_aux[i, :] = save_.gamma_aux
        save.signal_aux[i, :] = save_.signal_aux
        if not params.model_selection:
            save.gamma[i, :] = save_.gamma.reshape(-1)

    return save


def run_and_save(params):
    save = run(params=params)

    onp.savez(file='./experiments/results/{0}'.format(params.name),

              # randomness
              seed=params.seed,

              # data
              data=params.data,
              folds=params.folds,
              std_signal_region=params.std_signal_region,
              no_signal=params.no_signal,

              # parameters
              k=params.k,
              bins=params.bins,
              lower=params.lower,
              upper=params.upper,

              # ground truth
              mu_star=params.mu_star,
              sigma_star=params.sigma_star,
              sigma2_star=params.sigma2_star,
              lambda_star=params.lambda_star,

              # estimator
              gamma=save.gamma,
              gamma_aux=save.gamma_aux,
              signal_aux=save.signal_aux,

              # results
              bias=save.bias,
              estimates=save.estimates,
              coverage=save.coverage,
              width=save.width,
              cis=save.cis)
