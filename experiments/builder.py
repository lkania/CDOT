#######################################################
# Constants
#######################################################

PARAMETERS = ['lambda_star', 'lambda_star', 'mu_star', 'sigma2_star']
CIS_DELTA = ['lambda_hat0_delta', 'lambda_hat_delta', 'mu_hat_delta',
             'sigma2_hat_delta']
ESTIMATORS = ['lambda_hat0', 'lambda_hat', 'mu_hat', 'sigma2_hat']

#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np, random, jit
from jaxopt import AndersonAcceleration, FixedPointIteration
#######################################################
# Utilities
#######################################################
from functools import partial
from src.dotdic import DotDic
from src.load import load
from src.bin import proportions
#######################################################
# Basis
#######################################################
from src.basis import bernstein

#######################################################
# background methods
#######################################################
from src.background.bin import mle as bin_mle
#######################################################
# background transform
#######################################################
from src.transform import transform

#######################################################
# signal methods
#######################################################
from src.signal import mle as signal_mle

#######################################################
# signals
#######################################################
from jax.scipy.stats.norm import pdf as dnorm  # Gaussian/normal signal


#######################################################
# track stats
#######################################################


@jit
def normal(X, mu, sigma2):
    return dnorm(x=X.reshape(-1), loc=mu, scale=np.sqrt(sigma2)).reshape(-1)


#######################################################

# TODO: separate method and simulation parameters
def build_parameters(args):
    params = DotDic()
    params.seed = 0
    params.key = random.PRNGKey(seed=params.seed)

    #######################################################
    # Data parameters
    #######################################################
    params.cwd = args.cwd
    params.data_id = args.data_id
    params.data = '{0}/data/{1}/m_muamu.txt'.format(params.cwd,
                                                    params.data_id)

    #######################################################
    # load background data
    #######################################################
    print("Loading background data")
    params.background = DotDic()
    params.background.X = load(params.data)
    params.background.n = params.background.X.shape[0]
    print('Data source: {0} Size: {1}'.format(params.data, params.background.n))
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
    params.idx = random.permutation(key=params.key,
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
    # Background parameters
    #######################################################
    params.bins = args.bins  # high impact on jacobian computation for bin methods
    params.k = args.k  # high impact on jacobian computation for non-bin methods

    assert params.k is None or (params.bins >= (params.k + 1))

    params.ks = None
    params.bins_selection = None
    if args.ks is not None:
        assert len(args.ks) >= 2
        params.ks = args.ks
        params.bins_selection = int(
            np.ceil(params.bins * (args.bins_selection / 100) / 2))
        print("Model selection for k in {0}".format(params.ks))

    assert params.ks is None or (params.bins >= (max(params.ks) + 1))
    assert params.bins_selection is None or (
            params.bins_selection >= 0 and params.bins_selection <= 100)

    #######################################################
    # background transformation
    #######################################################
    params.rate = args.rate
    params.a = args.a
    params.b = args.b
    print('Data transformation. Support [{0},{1}] - rate {2} '.format(
        params.a,
        params.b,
        params.rate))

    trans, tilt_density, _ = transform(
        a=params.a, b=params.b, rate=params.rate)
    params.trans = trans
    params.tilt_density = tilt_density

    #######################################################
    # Basis
    #######################################################
    params.basis = bernstein

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
    params.tlower = params.trans(X=params.lower)
    params.tupper = params.trans(X=params.upper)

    #######################################################
    # allow 64 bits
    #######################################################
    params.dtype = np.float64

    #######################################################
    # numerical methods
    #######################################################
    params.tol = args.tol  # convergence criterion of iterative methods
    params.maxiter = args.maxiter
    # params.rcond = 1e-6  # thresholding for singular values

    #######################################################
    # signal
    #######################################################
    params.signal = DotDic()
    params.mixture = DotDic()

    params.signal.signal = normal
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
    # quantities of interest during simulation
    #######################################################
    params.true_parameters = [params.lambda_star, params.lambda_star,
                              params.mu_star, params.sigma2_star]
    params.estimators = ESTIMATORS
    params.ci_parameters = params.true_parameters
    params.cis = CIS_DELTA

    #######################################################
    # Background method
    #######################################################

    def fixpoint():
        params.optimizer += '_{0}'.format(args.fixpoint)
        match args.fixpoint:
            case 'normal':
                params.fixpoint = partial(FixedPointIteration,
                                          verbose=False,
                                          jit=True,
                                          implicit_diff=True,
                                          tol=params.tol,
                                          maxiter=params.maxiter)
            case 'anderson':
                params.fixpoint = partial(AndersonAcceleration,
                                          beta=1,
                                          history_size=5,
                                          mixing_frequency=1,
                                          verbose=False,
                                          jit=True,
                                          implicit_diff=True,
                                          tol=params.tol,
                                          maxiter=params.maxiter)
            case _:
                raise ValueError('Fixed point solver not supported')

    params.method = args.method
    params.optimizer = args.optimizer
    match params.method:
        case 'bin_mle':
            params.background.fit = bin_mle.fit
            match args.optimizer:
                case 'dagostini':
                    fixpoint()
                    params.background.optimizer = partial(
                        bin_mle.poisson_opt,
                        fixpoint=params.fixpoint,
                        _delta=bin_mle.dagostini,
                        tol=params.tol,
                        dtype=params.dtype)
                case 'normalized_dagostini':
                    fixpoint()
                    params.background.optimizer = partial(
                        bin_mle.poisson_opt,
                        fixpoint=params.fixpoint,
                        _delta=bin_mle.normalized_dagostini,
                        tol=params.tol,
                        dtype=params.dtype)
                case 'multinomial':
                    params.background.optimizer = partial(
                        bin_mle.multinomial_opt,
                        tol=params.tol,
                        maxiter=params.maxiter,
                        dtype=params.dtype)
                case 'mom':
                    params.background.optimizer = partial(
                        bin_mle.mom_opt,
                        tol=params.tol,
                        maxiter=params.maxiter,
                        dtype=params.dtype)
                case _:
                    raise ValueError('Optimizer not supported')

    print(
        "Method: {0} with {1} optimizer"
        "\n\tnumber of bins {2}"
        "\n\tbasis size {3}"
        "\n\tsignal region {4} standard deviations\n"
        .format(
            args.method,
            params.optimizer,
            params.bins,
            params.k,
            params.std_signal_region))

    #######################################################
    # Signal method
    #######################################################

    params.signal.fit = signal_mle.fit

    return params
