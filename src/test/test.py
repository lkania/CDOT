from jax import numpy as np, jit, grad
from functools import partial
# from src.bin import proportions
import jax
######################################################################
# local libraries
######################################################################
# from src.dotdic import DotDic
from src.test.builder import build as _build
from src.ci.delta import delta_ci, _delta_ci
from src.background.bin import test
from jax.scipy.stats.norm import cdf, ppf as icdf
from src import bin


def build(args):
	return _build(args)

# @partial(jit, static_argnames=['params'])
# def score_test(params, counts, n):
# 	counts = counts.reshape(-1)
# 	props = counts / n
#
# 	return method
