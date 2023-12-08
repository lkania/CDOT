from jax import numpy as np, random
#######################################################
# Utilities
#######################################################
from src.dotdic import DotDic
from src.load import load
from src.bin import proportions
from src.test.builder import build as _build


#######################################################

def load_background(args):
    params = DotDic()
    params.seed = args.seed
    params.key = random.PRNGKey(seed=params.seed)

    #######################################################
    # Data parameters
    #######################################################
    params.cwd = args.cwd
    params.data_id = args.data_id

    params.classifiers = args.classifiers

    #######################################################
    # load background data
    #######################################################
    params.path = '{0}/data/{1}'.format(params.cwd, params.data_id)
    print("Loading background data")
    params.background = DotDic()
    params.background.X = load(
        path='{0}/background/mass.txt'.format(params.path))
    params.background.c = DotDic()
    for classifier in params.classifiers:
        params.background.c[classifier] = load(
            path='{0}/background/{1}.txt'.format(params.path, classifier))

    print('Data source: {0} Size: {1}'.format(
        params.data_id, params.background.X.shape[0]))
    print("Data min={0} max={1}".format(
        np.round(np.min(params.background.X)),
        np.round(np.max(params.background.X))))

    #######################################################
    # Sampling
    #######################################################
    params.folds = args.folds
    params.sampling = DotDic()
    params.sampling.type = args.sampling_type
    params.sampling.size = args.sampling_size

    #######################################################
    # split background data
    #######################################################
    idx = random.permutation(
        key=params.key,
        x=np.arange(params.background.X.shape[0]),
        independent=False)

    match params.sampling.type:
        case 'independent':
            params.background.idxs = np.array_split(
                idx,
                indices_or_sections=params.folds)
        case 'subsample':
            params.background.idxs = random.choice(
                params.key, idx,
                shape=(params.folds,
                       params.sampling.size))
        case _:
            raise ValueError('Sampling type not supported')

    params.Xs = []
    params.cs = DotDic()
    for classifer in params.classifiers:
        params.cs[classifer] = []
    for i in range(params.folds):
        params.Xs.append(params.background.X[params.background.idxs[i]])
        for classifer in params.classifiers:
            params.cs[classifer].append(
                params.background.c[classifier][params.background.idxs[i]])

    return params


def load_signal(args, params):
    #######################################################
    # Fake signal parameters
    #######################################################
    # params.mu_star = args.mu_star
    # params.sigma_star = args.sigma_star
    # params.sigma2_star = np.square(params.sigma_star)
    params.lambda_star = args.lambda_star
    # print('Signal parameters: lambda {0} mu {1} sigma {2}'.format(
    #     params.lambda_star,
    #     params.mu_star,
    #     params.sigma_star
    # ))

    params.no_signal = (args.lambda_star < 1e-6)  # i.e. if lambda_star == 0

    #######################################################
    # amount of contamination
    #######################################################
    # params.std_signal_region = args.std_signal_region
    # params.lower = params.mu_star - params.std_signal_region * params.sigma_star
    # params.upper = params.mu_star + params.std_signal_region * params.sigma_star

    #######################################################
    # signal
    #######################################################
    params.signal = DotDic()
    # params.mixture = DotDic()

    # if params.no_signal:
    #     print("No signal will be added to the sample")
    #     params.mixture.X = params.background.X
    #     params.mixture.c = params.background.c

    if not params.no_signal:

        params.signal.X = load(
            path='{0}/signal/mass.txt'.format(params.path))

        params.signal.c = DotDic()
        for classifier in params.classifiers:
            params.signal.c[classifier] = load(
                path='{0}/signal/{1}.txt'.format(
                    params.path,
                    classifier))

        # prop, _ = proportions(params.signal.X,
        #                       np.array([params.lower]),
        #                       np.array([params.upper]))
        # prop = prop[0]
        # print(
        #     '{0:.2f}% of the loaded signal is contained in the signal region'
        #     .format(prop * 100))

        idx = random.permutation(
            key=params.key,
            x=np.arange(params.signal.X.shape[0]),
            independent=True)

        # params.mixture.X, params.mixture.c = add_signal(
        #     X=params.background.X,
        #     c=params.background.c)

        for i in range(params.folds):
            n = params.Xs[i].shape[0]
            n_signal = np.int32(n * params.lambda_star)
            idxs = random.choice(params.key, idx,
                                 shape=(n_signal,))
            signal_X = params.signal.X[idxs].reshape(-1)
            params.Xs[i] = np.concatenate((params.Xs[i], signal_X))

            for classifer in params.classifiers:
                signal_c = params.signal.c[classifer][idxs].reshape(-1)
                params.cs[classifer][i] = np.concatenate(
                    (params.cs[classifer][i], signal_c))

    return params


def filter(args, params):
    #######################################################
    # Classifier filter
    #######################################################

    # params.cutoff = args.cutoff
    # params.mixture.X_unfiltered = params.mixture.X
    # params.mixture.X = params.mixture.X[
    #     params.mixture.c >= params.cutoff].reshape(-1)

    min_ = params.Xs[0].shape[0]
    max_ = 0
    Xs = []
    for i in range(params.folds):
        cs = params.cs[args.classifier][i]
        X = params.Xs[i][cs >= args.cutoff].reshape(-1)
        X = random.permutation(
            key=params.key,
            x=X,
            independent=False)
        Xs.append(X)

        min_ = min(min_, X.shape[0])
        max_ = max(max_, X.shape[0])

    print("Simulation: {0} datasets of  size: [ {1} - {2} ] ".format(
        params.folds, min_, max_
    ))

    #######################################################
    # Modify args and extract parameters common to all the simulation
    #######################################################
    # args.upper = params.upper
    # args.lower = params.lower
    # method_params = _build(args)
    # params.ks = method_params.ks
    # params.tlower = method_params.tlower
    # params.tupper = method_params.tupper
    # params.bins = method_params.bins
    # params.basis = method_params.basis
    # params.trans = method_params.trans
    # params.optimizer = method_params.optimizer
    # params.method = method_params.method

    #######################################################
    # Define path for saving results of simulation
    #######################################################
    # params.path = '{0}/summaries/testing/{1}/{2}/{3}/{4}/'.format(
    #     params.cwd,
    #     params.data_id,
    #     'model_based' if method_params.model_signal else 'model_free',
    #     params.method,
    #     params.optimizer)
    # params.path += 'debias/' if method_params.debias else ''

    #######################################################
    # Print summary
    #######################################################

    # print("Method: {0} with {1} optimizer"
    #       "\n\tData transformation. Support [{2},{3}] - rate {4}"
    #       "\n\tNumber of bins {5}"
    #       "\n\tSignal region [{6},{7}]"
    #       "\n\tModel selection for k in {8}"
    #       "\n\t{9}"
    # .format(
    #     params.method,
    #     params.optimizer,
    #     method_params.a,
    #     method_params.b,
    #     method_params.rate,
    #     params.bins,
    #     params.lower,
    #     params.upper,
    #     params.ks,
    #     'model_based' if method_params.model_signal else 'model_free'))

    return Xs
