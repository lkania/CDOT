#######################################################
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np, random, jit

#######################################################
# Utilities
#######################################################
from src.dotdic import DotDic

#######################################################
# Basis
#######################################################
from src.basis import bernstein

#######################################################
# background methods
#######################################################
# from src.background.unbin import mom
# from src.background.bin import chi2 as bin_chi2
from src.background.bin import mle as bin_mle
# from src.background.bin import mom as bin_mom

#######################################################
# signal methods
#######################################################
from src.signal import mle as signal_mle

#######################################################
# signals
#######################################################
from jax.scipy.stats.norm import pdf as dnorm  # Gaussian/normal signal


@jit
def normal(X, mu, sigma2):
    return dnorm(x=X.reshape(-1), loc=mu, scale=np.sqrt(sigma2)).reshape(-1)


#######################################################

def build_parameters(args):
    params = DotDic()
    params.seed = 0
    params.key = random.PRNGKey(seed=params.seed)

    #######################################################
    # Data parameters
    #######################################################
    params.data_id = args.data_id
    params.data = '{0}/data/{1}/m_muamu.txt'.format(args.cwd, params.data_id)
    params.folds = 200

    #######################################################
    # Background parameters
    #######################################################
    params.k = args.k  # high impact on jacobian computation for non-bin methods
    params.bins = 100  # high impact on jacobian computation for bin methods

    #######################################################
    # parameters for background transformation
    #######################################################
    params.c = 0.003

    #######################################################
    # Basis
    #######################################################
    params.basis = bernstein

    #######################################################
    # Fake signal parameters
    #######################################################
    params.mu_star = 450
    params.sigma_star = 20
    params.sigma2_star = np.square(params.sigma_star)
    params.lambda_star = 0.01

    # if simulation is run without signal
    params.no_signal = args.no_signal
    if params.no_signal is True:
        params.lambda_star = 0
        # we don't modify sigma_star and mu_star so that the signal_region stays the same
        # regardless of lambda

    #######################################################
    # amount of contamination
    #######################################################
    params.std_signal_region = args.std_signal_region
    params.lower = params.mu_star - params.std_signal_region * params.sigma_star
    params.upper = params.mu_star + params.std_signal_region * params.sigma_star

    #######################################################
    # allow 64 bits
    #######################################################
    params.dtype = np.float64

    #######################################################
    # numerical methods
    #######################################################
    params.tol = 1e-6  # convergence criterion of iterative methods
    params.maxiter = 10000
    # params.rcond = 1e-6  # thresholding for singular values

    #######################################################
    # signal
    #######################################################
    params.signal.signal = normal

    #######################################################
    # quantities of interest during simulation
    #######################################################
    params.true_parameters = [params.lambda_star, params.lambda_star, params.mu_star, params.sigma2_star]
    params.parameters = ['lambda_hat0', 'lambda_hat', 'mu_hat', 'sigma2_hat']
    params.ci_parameters = params.true_parameters
    params.cis = ['lambda_hat0_delta', 'lambda_hat_delta', 'mu_hat_delta', 'sigma2_hat_delta']

    #######################################################
    # Background method
    #######################################################
    match args.method:
        case 'bin_mle':
            params.background.fit = bin_mle.fit
        # case 'bin_mom':
        #     params.background = bin_mom
        # case 'bin_chi2':
        #     params.background = bin_chi2
        # case 'mom':
        #     params.background = mom

    # match args.nnls:
    #     case 'conic_cvx':
    #         params.nnls = conic_cvx_nnls
    #     case 'pg_jaxopt':
    #         params.nnls = pg_jaxopt_nnls
    #     case 'pg_jax':
    #         params.nnls = pg_jax_nnls
    #     case 'lawson_scipy':
    #         params.nnls = lawson_scipy_nnls
    #     case 'lawson_jax':
    #         params.nnls = lawson_jax_nnls

    #######################################################
    # Signal method
    #######################################################

    params.signal.fit = signal_mle.fit

    #######################################################
    # Name
    #######################################################

    # the name contains the following information
    # - optimizer
    # - number of background parameters
    # - dataset id
    # - std for signal region
    # - presence of signal
    if args.nnls == 'None':
        params.name = args.method
    else:
        params.name = '_'.join([args.method, args.nnls])

    params.name = '_'.join(
        [params.name,
         str(params.k),
         str(params.std_signal_region),
         str(params.no_signal),
         str(params.data_id),
         str(params.folds)])

    # sanity checks
    assert (params.bins > params.k + 1)

    return params