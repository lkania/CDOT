#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np, random
#######################################################
# Utilities
#######################################################
from src.dotdic import DotDic
from src.load import load
from src.bin import proportions
from src.test.builder import build as _build


#######################################################

def build(args):
    params = DotDic()
    params.seed = args.seed
    params.key = random.PRNGKey(seed=params.seed)

    #######################################################
    # Data parameters
    #######################################################
    params.cwd = args.cwd
    params.data_id = args.data_id
    params.data = '{0}/data/{1}/m_muamu.txt'.format(
        params.cwd,
        params.data_id)

    #######################################################
    # load background data
    #######################################################
    print("Loading background data")
    params.background = DotDic()
    params.background.X = load(params.data)
    params.background.n = params.background.X.shape[0]
    print('Data source: {0} Size: {1}'.format(
        params.data, params.background.n))
    print("Minimum value of the data {0}".format(
        np.round(np.min(params.background.X)), 2))
    print("Maximum value of the data {0}".format(
        np.round(np.max(params.background.X)), 2))

    #######################################################
    # Sampling
    #######################################################
    params.folds = args.folds
    params.sampling = DotDic()
    params.sampling.type = args.sampling_type
    params.sampling.size = args.sampling_size

    #######################################################
    # split data
    #######################################################
    params.idx = random.permutation(
        key=params.key,
        x=np.arange(params.background.n),
        independent=True)

    match params.sampling.type:
        case 'independent':
            params.idxs = np.array_split(params.idx,
                                         indices_or_sections=params.folds)
            params.sampling.size = params.background.X[params.idxs[0]].shape[0]
        case 'subsample':
            params.idxs = random.choice(params.key, params.idx,
                                        shape=(params.folds,
                                               params.sampling.size))
        case _:
            raise ValueError('Sampling type not supported')

    print("{0}: Number {1} Size {2}".format(
        params.sampling.type,
        params.folds,
        params.sampling.size))
    #######################################################
    # Fake signal parameters
    #######################################################
    params.mu_star = args.mu_star
    params.sigma_star = args.sigma_star
    params.sigma2_star = np.square(params.sigma_star)
    params.lambda_star = args.lambda_star
    print('Signal parameters: lambda {0} mu {1} sigma {2}'.format(
        params.lambda_star,
        params.mu_star,
        params.sigma_star
    ))

    params.no_signal = (args.lambda_star < 1e-6)  # i.e. if lambda_star == 0

    #######################################################
    # amount of contamination
    #######################################################
    params.std_signal_region = args.std_signal_region
    params.lower = params.mu_star - params.std_signal_region * params.sigma_star
    params.upper = params.mu_star + params.std_signal_region * params.sigma_star

    #######################################################
    # signal
    #######################################################
    params.signal = DotDic()
    params.mixture = DotDic()

    match args.signal:
        case 'normal':
            params.signal.sample = \
                lambda n: params.mu_star + params.sigma_star * random.normal(
                    params.key, shape=(n,))
        case 'file':
            path = '{0}/data/{1}/signal.txt'.format(
                params.cwd,
                params.data_id)
            signal = load(path)
            prop, _ = proportions(signal,
                                  np.array([params.lower]),
                                  np.array([params.upper]))
            prop = prop[0]
            print(
                '{0:.2f}% of the loaded signal is contained in the signal region'
                .format(prop * 100))
            params.signal.sample = \
                lambda n: random.choice(params.key, signal, shape=(n,))

    if params.no_signal:
        print("No signal will be added to the sample")
        params.mixture.X = params.background.X
    else:
        n_signal = np.int32(params.background.n * params.lambda_star)
        signal = params.signal.sample(n_signal)
        signal = signal.reshape(-1)
        params.mixture.X = np.concatenate((params.background.X,
                                           signal))

    params.mixture.n = params.mixture.X.shape[0]

    #######################################################
    # Modify args and extract parameters common to all the simulation
    #######################################################
    args.upper = params.upper
    args.lower = params.lower
    method_params = _build(args)
    params.ks = method_params.ks
    params.tlower = method_params.tlower
    params.tupper = method_params.tupper
    params.bins = method_params.bins
    params.basis = method_params.basis
    params.trans = method_params.trans
    params.optimizer = method_params.optimizer
    params.method = method_params.method

    #######################################################
    # Define path for saving results of simulation
    #######################################################
    params.path = '{0}/summaries/testing/{1}/{2}/{3}/{4}/'.format(
        params.cwd,
        params.data_id,
        'model_based' if method_params.model_signal else 'model_free',
        params.method,
        params.optimizer)

    #######################################################
    # Print summary
    #######################################################

    print("Method: {0} with {1} optimizer"
          "\n\tData transformation. Support [{2},{3}] - rate {4}"
          "\n\tNumber of bins {5}"
          "\n\tSignal region [{6},{7}]"
          "\n\tModel selection for k in {8}"
          "\n\t{9}"
    .format(
        params.method,
        params.optimizer,
        method_params.a,
        method_params.b,
        method_params.rate,
        params.bins,
        params.lower,
        params.upper,
        params.ks,
        'model_based' if method_params.model_signal else 'model_free'))

    return params
