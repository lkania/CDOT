from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()

    parser.add_argument("--method", type=str, default='bin_mle')

    parser.add_argument("--folds", default=500, type=int)
    parser.add_argument("--data_id", default='50', type=str)
    parser.add_argument("--sample_split", default=False, type=bool)

    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--bins", default=100, type=int)

    parser.add_argument("--std_signal_region", default=3, type=float)

    parser.add_argument("--mu_star", default=450, type=int)
    parser.add_argument("--sigma_star", default=20, type=int)
    parser.add_argument("--lambda_star", default=0.01, type=float)

    parser.add_argument("--rate", default=0.003, type=float)
    parser.add_argument("--a", default=250, type=float)
    parser.add_argument("--b", default=0, type=float)

    args, _ = parser.parse_known_args()

    return args
