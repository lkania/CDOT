# In order to take advantage of jit, run each method independently
# so that we avoid recompilation of methods that depend on params

import localize  # change the cwd to the root of the project
from experiments.evaluate import run_and_save
from experiments.parameters import build_parameters
from experiments.parser import parse

#######################################################
# Read arguments
#######################################################


args = parse()
args.cwd = '..'
params = build_parameters(args)

print('Name of the method: {0}'.format(params.name))

print('Truth. lambda_star {0} - mu_star {1} - sigma_star {2}'.format(params.lambda_star,
                                                                     params.mu_star,
                                                                     params.sigma_star))

print('Transformation. [{0},{1}] - rate {2} '.format(params.a, params.b, params.rate))

run_and_save(params=params)

print('\n{0} finished'.format(params.name))
