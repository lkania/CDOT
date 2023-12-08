from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()

    ######################################################################
    # Method parameters
    ######################################################################

    parser.add_argument("--seed", type=int, default=0)

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

    parser.add_argument(
        "--rate", default=0.003, type=float)
    parser.add_argument(
        "--a", default=201, type=float)
    parser.add_argument(
        '--b',
        default=None,
        type=lambda x: None if x == 'None' else float(x))
    parser.add_argument(
        "--k",
        help='Order of basis',
        default=None,
        type=lambda x: None if x == 'None' else int(x))

    parser.add_argument(
        '--ks',
        nargs='+',
        help='Order range for model selection',
        default=[1, 2],
        type=int)

    parser.add_argument(
        '--bins_selection',
        help='Percentage of bins used for model selection',
        default=20,
        type=float)

    parser.add_argument("--bins",
                        default=100,
                        type=int)

    parser.add_argument(
        "--model_signal",
        help='The signal is modelled by a normal distribution with unknown parameters',
        default=False,
        type=lambda x: True if x == 'True' or x == 'true' else False)

    parser.add_argument(
        "--debias",
        help='Split the sample and de-bias',
        default=False,
        type=lambda x: True if x == 'True' or x == 'true' else False)

    parser.add_argument(
        "--cutoff",
        help='Threshold for classifier',
        default=0.5,
        type=float)

    parser.add_argument(
        "--transformed_cutoff",
        help='Use transformed cutoff',
        default=True,
        type=lambda x: True if x == 'True' or x == 'true' else False)

    ######################################################################
    # Simulation parameters
    ######################################################################
    parser.add_argument("--cwd", type=str, default='..')

    parser.add_argument("--folds",
                        default=3,
                        type=int)

    parser.add_argument("--sampling_type",
                        default="subsample",
                        type=str)

    parser.add_argument("--sampling_size",
                        default=15000,
                        type=int)

    parser.add_argument("--data_id",
                        default='3b',
                        type=str)

    parser.add_argument("--std_signal_region",
                        default=1.5,
                        type=float)
    parser.add_argument("--mu_star",
                        default=395,
                        type=float)
    parser.add_argument("--sigma_star",
                        default=20,
                        type=float)
    parser.add_argument("--lambda_star",
                        default=0.0,
                        type=float)
    parser.add_argument("--signal",
                        default='file',
                        type=str)

    args, _ = parser.parse_known_args()

    args.classifiers = ["tclass", "class"]
    args.lower = args.mu_star - args.std_signal_region * args.sigma_star
    args.upper = args.mu_star + args.std_signal_region * args.sigma_star

    return args
