from jax import numpy as np, random, jit, jacrev, grad
from jaxopt import AndersonAcceleration, FixedPointIteration

#######################################################
# Utilities
#######################################################
from functools import partial
from src.dotdic import DotDic

#######################################################
# background methods
#######################################################
from src.background.bin import mle as bin_mle


def build(args):
	#######################################################
	# init sub-dictionaries
	#######################################################
	params = DotDic()
	params.hash = args.hash
	params.background = DotDic()

	#######################################################
	# Background estimation parameters
	#######################################################
	# high impact on jacobian computation for bin methods
	params.from_ = args.from_
	params.to_ = args.to_
	assert params.from_.shape[0] == params.to_.shape[0]
	params.bins = len(params.from_)

	# high impact on jacobian computation for non-bin methods
	assert args.k is not None
	params.k = int(args.k)

	# Certify that there will be enough data-points to fit the background density
	assert params.k <= (params.bins + 1)

	#######################################################
	# Basis
	#######################################################
	params.basis = args.basis

	#######################################################
	# signal region
	#######################################################
	params.lower = args.lower
	params.upper = args.upper

	#######################################################
	# numerical methods
	#######################################################
	params.tol = args.tol  # convergence criterion of iterative methods
	params.maxiter = args.maxiter

	#######################################################
	# optimizer used in background estimation
	#######################################################

	params.fixpoint = args.fixpoint
	match params.fixpoint:
		case 'anderson':
			fixpoint = partial(AndersonAcceleration,
							   beta=1,
							   history_size=5,
							   mixing_frequency=1,
							   verbose=False,
							   jit=True,
							   implicit_diff=True,
							   tol=params.tol,
							   maxiter=params.maxiter)
		case 'normal' | _:  # default choice
			fixpoint = partial(FixedPointIteration,
							   verbose=False,
							   jit=True,
							   implicit_diff=True,
							   tol=params.tol,
							   maxiter=params.maxiter)

	params.grad_op = grad
	params.method = args.method
	params.optimizer = args.optimizer
	match params.method:
		case 'bin_mle':
			match args.optimizer:
				# case 'normalized_dagostini':
				# params.background.optimizer = partial(
				# 	bin_mle.poisson_opt,
				# 	fixpoint=fixpoint,
				# 	_delta=bin_mle.normalized_dagostini,
				# 	tol=params.tol)
				case 'dagostini' | _:  # default choice
					params.background.optimizer = partial(
						bin_mle.poisson_opt,
						fixpoint=fixpoint,
						_delta=bin_mle.dagostini,
						tol=params.tol)

			bin_mle.preprocess(params)

		# case 'multinomial':
		# 	params.background.optimizer = partial(
		# 		bin_mle.multinomial_opt,
		# 		tol=params.tol,
		# 		maxiter=params.maxiter,
		# 		dtype=params.dtype)
		# case 'mom':
		# 	params.background.optimizer = partial(
		# 		bin_mle.mom_opt,
		# 		tol=params.tol,
		# 		maxiter=params.maxiter,
		# 		dtype=params.dtype)
		# case _:
		# 	raise ValueError('Optimizer not supported')

	return params
