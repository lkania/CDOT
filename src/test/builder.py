#######################################################
# Utilities
#######################################################
from functools import partial

from jax import numpy as np
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt.projection import projection_polyhedron, projection_simplex
#######################################################
# background methods
#######################################################
from src.background.bin import test as bin_mle
from src.background.unbin import test as unbin_mle
from src.dotdic import DotDic


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

	init_lambda = 0
	init_gamma = ((np.zeros(shape=(params.k + 1), dtype=args.float) + 1) / (
			params.k + 1)).reshape(-1)
	params.background.init_lambda = init_lambda
	params.background.init_gamma = init_gamma

	# compute integrals
	int_omega = params.basis.int_omega(k=params.k).reshape(-1)
	assert not np.isnan(int_omega).any()
	assert np.all(int_omega > params.tol)

	int_signal = params.basis.integrate(k=params.k,
										a=params.lower,
										b=params.upper).reshape(-1)
	assert not np.isnan(int_signal).any()
	assert np.all(int_signal > 0)

	int_control = int_omega - int_signal
	assert np.all(int_control > 0)

	params.background.int_signal = int_signal
	params.background.int_omega = int_omega
	params.background.int_control = int_control

	params.method = args.method
	params.optimizer = args.optimizer
	match params.method:
		case 'unbin_mle':

			match args.optimizer:
				case 'density_with_opt_lambda':
					# only for Bernstein basis
					projection = partial(
						projection_simplex,
						value=params.background.init_gamma.shape[0])
					params.estimate = partial(
						unbin_mle.constrained_opt_with_opt_lambda,
						maxiter=params.maxiter,
						tol=params.tol,
						projection=projection,
						init_gamma=params.background.init_gamma
					)

				case 'density':
					lambda_lowerbound = 0  # TODO: can produce upper bias
					lambda_upperbound = 0.5
					gamma_lowerbound = 0

					zero = np.array([0])
					lambda_upperbound = np.array([lambda_upperbound])

					n_params = params.background.init_gamma.reshape(-1).shape[0]

					# Equality constraint
					# force the basis to integrate to 1 over the omega domain
					A_ = (
						np.concatenate((int_omega, zero)).reshape(-1)).reshape(
						1, -1)
					b_ = np.array([1.0])

					# If using projection_polyhedron
					# then use the following inequality constraints
					G = -1 * np.eye(n_params + 2, n_params + 1)
					G = G.at[-1].set(np.zeros(n_params + 1, ).at[-1].set(1))
					lower_bounds = np.zeros((n_params + 1,)) + gamma_lowerbound
					lower_bounds = lower_bounds.at[-1].set(lambda_lowerbound)
					h = np.concatenate((lower_bounds, lambda_upperbound))

					projection = partial(projection_polyhedron,
										 hyperparams_proj=(A_, b_, G, h),
										 check_feasible=False)

					params.estimate = partial(
						unbin_mle.constrained_opt,
						maxiter=params.maxiter,
						tol=params.tol,
						projection=projection,
						init_lambda=params.background.init_lambda,
						init_gamma=params.background.init_gamma
					)

				case 'normalized_dagostini':
					params.estimate = partial(
						unbin_mle.EM_opt,
						init_gamma=init_gamma,
						fixpoint=fixpoint,
						update=partial(unbin_mle.normalized_dagostini,
									   tol=params.tol,
									   int_omega=params.background.int_omega,
									   int_control=params.background.int_control),
						int_control=params.background.int_control,
						int_omega=params.background.int_omega)

				case 'dagostini' | _:  # default choice
					params.estimate = partial(
						unbin_mle.EM_opt,
						init_gamma=init_gamma,
						fixpoint=fixpoint,
						update=partial(unbin_mle.dagostini,
									   tol=params.tol,
									   int_control=params.background.int_control),
						int_control=params.background.int_control,
						int_omega=params.background.int_omega)

			return partial(unbin_mle.efficient_test, params=params)

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
			M = params.basis.integrate(params.k,
									   params.from_,
									   params.to_)  # n_bins x n_parameters
			assert not np.isnan(M).any()

			int_control_ = np.sum(M, axis=0).reshape(-1)
			assert not np.isnan(int_control_).any()
			assert np.all(int_control_ > params.tol)
			assert np.all(np.abs(int_control - int_control_) < params.tol)

			params.background.M = M

			match args.optimizer:
				case 'normalized_dagostini':
					params.estimate = partial(
						bin_mle.EM_opt,
						fixpoint=fixpoint,
						update=partial(bin_mle.normalized_dagostini,
									   M=params.background.M,
									   int_control=params.background.int_control,
									   int_omega=params.background.int_omega,
									   tol=params.tol),
						init_gamma=params.background.init_gamma,
						int_control=params.background.int_control,
						int_omega=params.background.int_omega)
				case 'multinomial':
					params.estimate = partial(
						bin_mle.multinomial_opt,
						maxiter=params.maxiter,
						tol=params.tol)
				case 'dagostini' | _:  # default choice
					params.estimate = partial(
						bin_mle.EM_opt,
						fixpoint=fixpoint,
						update=partial(bin_mle.dagostini,
									   M=params.background.M,
									   int_control=params.background.int_control,
									   tol=params.tol),
						init_gamma=params.background.init_gamma,
						int_control=params.background.int_control,
						int_omega=params.background.int_omega)

			return partial(bin_mle.delta_method_test, params=params)

		case _:
			raise ValueError('Method not supported')
