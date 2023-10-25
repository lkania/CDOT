from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()

    parser.add_argument("--cwd", type=str, default='..')

    parser.add_argument("--method",
                        type=str,
                        default='bin_mle')

    parser.add_argument("--optimizer",
                        type=str,
                        default='dagostini')

    parser.add_argument("--fixpoint",
                        type=str,
                        default='normal')

    parser.add_argument("--maxiter",
                        type=int,
                        default=100)

    parser.add_argument("--tol",
                        type=float,
                        default=1e-6)

    parser.add_argument("--folds",
                        default=3,
                        type=int)

    parser.add_argument("--sampling_type",
                        default="subsample",
                        type=str)

    parser.add_argument("--sampling_size",
                        default=3000,
                        type=int)

    parser.add_argument("--data_id",
                        default='4b',
                        type=str)

    parser.add_argument("--k",
                        help='Order of basis',
                        default=None,
                        type=lambda x: None if x == 'None' else int(x))

    parser.add_argument('--ks',
                        nargs='+',
                        help='Order range for model selection',
                        default=[4, 10, 15, 20, 25, 30],
                        type=int)

    parser.add_argument('--bins_selection',
                        help='Percentage of bins used for model selection',
                        default=20,
                        type=float)

    parser.add_argument("--bins", default=100, type=int)

    parser.add_argument("--std_signal_region",
                        default=2.5,
                        type=float)
    parser.add_argument("--mu_star", default=395.8171, type=float)
    parser.add_argument("--sigma_star", default=20.33321, type=float)
    parser.add_argument("--lambda_star", default=0.051, type=float)
    parser.add_argument("--signal", default='file', type=str)

    parser.add_argument("--rate", default=0.003, type=float)
    parser.add_argument("--a", default=201, type=float)
    parser.add_argument('--b',
                        default=None,
                        type=lambda x: None if x == 'None' else float(x))

    args, _ = parser.parse_known_args()

    return args
