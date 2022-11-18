# In order to take advantage of jit, run each method independently
# so that we avoid recompilation of methods that depend on params

import localize  # change the cwd to the root of the project
from experiments.evaluate import run_and_save
from experiments.parameters import build_parameters
from argparse import ArgumentParser

#######################################################
# Read arguments
#######################################################

parser = ArgumentParser()

parser.add_argument("--method", type=str, default='bin_mle')
parser.add_argument("--k", type=int, default=20)

parser.add_argument("--std_signal_region", default=3, type=float)
parser.add_argument("--data_id", default='50', type=str)

parser.add_argument("--mu_star", default=450, type=int)
parser.add_argument("--sigma_star", default=20, type=int)
parser.add_argument("--lambda_star", default=0.01, type=float)

parser.add_argument("--rate", default=0.003, type=float)
parser.add_argument("--a", default=250, type=float)
parser.add_argument("--b", default=0, type=float)

args = parser.parse_args()
args.cwd = '..'

params = build_parameters(args)

print('Name of the method: {0}'.format(params.name))

print('Truth. lambda_star {0} - mu_star {1} - sigma_star {2}'.format(params.lambda_star,
                                                                     params.mu_star,
                                                                     params.sigma_star))

print('Transformation. [{0},{1}] - rate {2} '.format(params.a, params.b, params.rate))

run_and_save(params=params)

print('\n{0} finished'.format(params.name))
