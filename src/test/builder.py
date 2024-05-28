from jax import grad, numpy as np
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

	# high impact on jacobian computation for non-bin methods
	assert args.k is not None
	params.k = int(args.k)

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

	params.background.init_lambda = 0.01
	params.background.init_gamma = ((np.zeros(shape=(params.k + 1),
											  dtype=args.float) + 1) / (
											params.k + 1)).reshape(-1)

	params.method = args.method
	params.optimizer = args.optimizer
	match params.method:
		case 'bin_mle':
			# high impact on jacobian computation for bin methods
			params.from_ = args.from_
			params.to_ = args.to_
			assert params.from_.shape[0] == params.to_.shape[0]
			params.bins = len(params.from_)
			# Certify that there will be enough data-points
			# to fit the background density
			assert params.k <= (params.bins + 1)

			# compute key quantities
			int_omega = params.basis.int_omega(k=params.k)
			assert not np.isnan(int_omega).any()
			assert np.all(int_omega > params.tol)

			M = params.basis.integrate(params.k,
									   params.from_,
									   params.to_)  # n_bins x n_parameters
			assert not np.isnan(M).any()

			int_control = np.sum(M, axis=0).reshape(-1, 1)
			assert not np.isnan(int_control).any()
			assert np.all(int_control > params.tol)

			params.background.int_omega = int_omega
			params.background.M = M
			params.background.int_control = int_control

			match args.optimizer:
				case 'poisson':
					params.background.optimizer = partial(
						bin_mle.poisson_opt,
						fixpoint=fixpoint,
						_delta=partial(bin_mle.normalized_dagostini,
									   tol=params.tol),
						tol=params.tol)
				case 'multinomial':
					params.background.optimizer = partial(
						bin_mle.multinomial_opt,
						maxiter=params.maxiter,
						tol=params.tol)
				case 'dagostini' | _:  # default choice
					params.background.optimizer = partial(
						bin_mle.poisson_opt,
						fixpoint=fixpoint,
						_delta=partial(bin_mle.dagostini,
									   tol=params.tol),
						tol=params.tol)

			params.background.estimate_lambda = partial(
				params.background.optimizer,
				init_lambda=params.background.init_lambda,
				init_gamma=params.background.init_gamma,
				M=params.background.M,
				int_control=params.background.int_control,
				int_omega=params.background.int_omega)

		# bin_mle.preprocess(params)

		case 'unbin_mle':
			raise ValueError('Method not supported')

	return params
