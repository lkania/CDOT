from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()

    parser.add_argument("--cwd", type=str, default='..')

    parser.add_argument("--method", type=str, default='bin_mle')
    parser.add_argument("--optimizer", type=str, default='poisson')
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-4)

    parser.add_argument("--folds", default=50, type=int)
    parser.add_argument("--sampling_type", default="subsample", type=str)
    parser.add_argument("--sampling_size", default=3000, type=int)

    parser.add_argument("--data_id", default='3b', type=str)
    parser.add_argument("--sample_split", default=False, type=bool)

    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--bins", default=50, type=int)

    parser.add_argument("--std_signal_region", default=3, type=float)
    parser.add_argument("--mu_star", default=395.8171, type=float)
    parser.add_argument("--sigma_star", default=20.33321, type=float)
    parser.add_argument("--lambda_star", default=0, type=float)
    parser.add_argument("--signal", default='file', type=str)

    parser.add_argument("--rate", default=0.003, type=float)
    parser.add_argument("--a", default=201, type=float)
    parser.add_argument("--b", default=0, type=float)

    args, _ = parser.parse_known_args()

    return args
