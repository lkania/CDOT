from src.load import load
from src.transform import transform
from src.random import add_signal
import jax.random as random
import jax.numpy as np
import numpy as onp
from tqdm import tqdm
from src.dotdic import DotDic
from src.ci.delta import compute_cis


def _evaluate(X, params):
    add_signal(X=X, params=params)

    trans, tilt_density = transform(params.X)

    estimand = params.estimator.fit(X=trans(params.X),
                                    lower=trans(params.lower),
                                    upper=trans(params.upper),
                                    params=params)

    params.trans = trans
    params.tilt_density = tilt_density

    compute_cis(params=params, model=estimand)

    return estimand


def evaluate(X, params):
    estimand = _evaluate(X, params)

    #######################################################
    # save results
    #######################################################
    save = DotDic()
    save.bias = onp.zeros((len(params.parameters),), dtype=params.dtype)
    save.estimates = onp.zeros((len(params.parameters),), dtype=params.dtype)
    save.coverage = onp.zeros((len(params.cis),), dtype=onp.bool_)
    save.width = onp.zeros((len(params.cis),), dtype=params.dtype)
    save.cis = onp.zeros((len(params.cis), 2), dtype=params.dtype)
    save.gamma = estimand.gamma
    save.gamma_error = estimand.gamma_error
    save.signal_error = estimand.signal_error
    save.signal_region = (params.lower, params.upper)

    # bias
    for i in np.arange(len(params.parameters)):
        estimated_parameter = estimand[params.parameters[i]]
        save.bias[i] = estimated_parameter - params.true_parameters[i]
        save.estimates[i] = estimated_parameter

    # coverage
    for i in np.arange(len(params.cis)):
        parameter = params.ci_parameters[i]
        ci = estimand[params.cis[i]]
        save.coverage[i] = (ci[0] <= parameter) and (parameter <= ci[1])
        save.width[i] = ci[1] - ci[0]
        save.cis[i, :] = ci

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
    idxs = np.array(np.split(idx, indices_or_sections=params.folds))

    #######################################################
    # print info
    #######################################################
    print('\n{0}'.format(params.name))
    print('Data source: {0}'.format(params.data))
    print('Using {0} datasets'.format(params.folds))
    print('Datasets have {0} examples'.format(X_[idxs[0, :]].shape[0]))

    #######################################################
    # quantities of interest during simulation
    #######################################################
    params.true_parameters = [params.lambda_star,
                              params.lambda_star,
                              params.mu_star,
                              np.square(params.sigma_star)]
    params.parameters = ['lambda_hat0',
                         'lambda_hat',
                         'mu_hat',
                         'sigma2_hat']
    params.ci_parameters = params.true_parameters
    params.cis = ['lambda_hat0_delta',
                  'lambda_hat_delta',
                  'mu_hat_delta',
                  'sigma2_hat_delta']

    save = DotDic()
    save.bias = onp.zeros((params.folds, len(params.parameters)),
                          dtype=params.dtype)
    save.coverage = onp.zeros((params.folds, len(params.cis)), dtype=onp.bool_)
    save.width = onp.zeros((params.folds, len(params.cis)), dtype=params.dtype)
    save.estimates = onp.zeros((params.folds, len(params.parameters)),
                               dtype=params.dtype)
    save.cis = onp.zeros((params.folds, len(params.cis), 2), dtype=params.dtype)
    save.gamma = onp.zeros((params.folds, params.k + 1), dtype=params.dtype)
    save.gamma_error = onp.zeros((params.folds,), dtype=params.dtype)
    save.signal_error = onp.zeros((params.folds,), dtype=params.dtype)
    save.signal_region = onp.zeros((params.folds, 2), dtype=params.dtype)

    for i in tqdm(range(params.folds), dynamic_ncols=True):
        save_ = evaluate(X=X_[idxs[i, :]], params=params)
        save.bias[i, :] = save_.bias
        save.coverage[i, :] = save_.coverage
        save.width[i, :] = save_.width
        save.estimates[i, :] = save_.estimates
        save.cis[i, :, :] = save_.cis
        save.gamma[i, :] = save_.gamma.reshape(-1)
        save.gamma_error[i] = save_.gamma_error
        save.signal_error[i] = save_.signal_error
        save.signal_region[i, :] = save_.signal_region

    return save


def run_and_save(params):
    save = run(params=params)

    onp.savez(file='./results/{0}'.format(params.name),

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
              signal_region=save.signal_region,

              # ground truth
              mu_star=params.mu_star,
              sigma_star=params.sigma_star,
              lambda_star=params.lambda_star,

              # estimator
              gamma=save.gamma,
              gamma_error=save.gamma_error,
              signal_error=save.signal_error,

              # results
              bias=save.bias,
              estimates=save.estimates,
              coverage=save.coverage,
              width=save.width,
              cis=save.cis)
