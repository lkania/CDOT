# In order to take advantage of jit, run each method independently
# so that we avoid recompilation of methods that depend on params

# change the cwd to the root of the project
# otherwise the local packages cannot be found
import localize
from experiments.evaluate import run_and_save
from experiments.builder import build_parameters
from experiments.parser import parse

#######################################################
# Read arguments
#######################################################


args = parse()

params = build_parameters(args)

run_and_save(params=params)

print('\n{0} finished'.format(params.name))
