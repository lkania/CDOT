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
parser.add_argument("--std_signal_region", type=int, default=3)
parser.add_argument("--no_signal", type=str, default='False')
parser.add_argument("--nnls", default='None', type=str)
parser.add_argument("--data_id", default=50, type=int)
args = parser.parse_args()
args.no_signal = (args.no_signal.lower() == 'true')
args.cwd = '..'

params = build_parameters(args)

print('Name of the method: {0}'.format(params.name))
run_and_save(params=params)
print('{0} finished'.format(params.name))
