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

	match args.fixpoint:
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

	# init_lambda = 0
	init_gamma = np.full(shape=(params.k + 1,), fill_value=1 / (params.k + 1))
	init_gamma = init_gamma.reshape(-1)

	assert np.abs(np.sum(init_gamma) - 1) < params.tol


	# assert that each basis element integrates to one over omega
	int_omega = params.basis.integrate(k=params.k, a=0, b=1).reshape(-1)
	assert np.all(int_omega > 0)
	assert np.max(np.abs(int_omega - 1)) < params.tol
	int_omega = 1

	int_signal = params.basis.integrate(k=params.k,
										a=params.lower,
										b=params.upper).reshape(-1)
	assert not np.isnan(int_signal).any()
	assert np.all(int_signal > 0)

	int_control = int_omega - int_signal
	assert np.all(int_control > 0)

	# check that the basis properly integrates to one over omega
	int_control_ = params.basis.integrate(
		k=params.k,
		a=0,
		b=params.lower).reshape(-1)
	int_control_ += params.basis.integrate(
		k=params.k,
		a=params.upper,
		b=1).reshape(-1)
	assert np.max(np.abs(int_control_ - int_control)) < params.tol
	assert np.max(np.abs(int_signal + int_control_ - int_omega)) < params.tol

	params.method = args.method
	params.optimizer = args.optimizer
	match params.method:
		case 'unbin_mle':

			match args.optimizer:
				case 'constrained_opt':

					params.estimate = partial(
						unbin_mle.constrained_opt_with_opt_lambda,
						loss=partial(unbin_mle.loss_with_opt_lambda,
									 int_control=int_control,
									 tol=params.tol),
						maxiter=params.maxiter,
						tol=params.tol,
						projection=projection_simplex,
						init_gamma=init_gamma,
						int_control=int_control
					)

				case 'density':
					lambda_lowerbound = 0
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
									   int_control=int_control),
						int_control=int_control
					)

				case 'dagostini':
					params.estimate = partial(
						unbin_mle.EM_opt,
						init_gamma=init_gamma,
						fixpoint=fixpoint,
						update=partial(unbin_mle.dagostini,
									   tol=params.tol,
									   int_control=int_control),
						int_control=int_control,
						int_omega=int_omega)
				case _:
					raise ValueError('Method not supported')

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

			# compute integral of each basis over the control region bins
			M = params.basis.integrate(params.k,
									   params.from_,
									   params.to_)  # n_bins x n_parameters
			assert not np.isnan(M).any()

			# check that adding integrals over bins on the control region
			# recovers the integral over the whole control region
			int_control_ = np.sum(M, axis=0).reshape(-1)
			assert not np.isnan(int_control_).any()
			assert np.max(np.abs(int_control - int_control_)) < params.tol

			match args.optimizer:

				case 'constrained_opt':
					params.estimate = partial(
						bin_mle.constrained_opt,
						loss=partial(bin_mle.loss,
									 M=M,
									 int_control=int_control,
									 tol=params.tol),
						init_gamma=init_gamma,
						int_control=int_control,
						projection=projection_simplex,
						maxiter=params.maxiter,
						tol=params.tol)
				case 'dagostini':
					params.estimate = partial(
						bin_mle.EM_opt,
						fixpoint=fixpoint,
						update=partial(bin_mle.dagostini,
									   M=M,
									   int_control=int_control,
									   tol=params.tol),
						init_gamma=init_gamma,
						int_control=int_control,
						int_omega=int_omega)
				case 'normalized_dagostini':  # default choice
					params.estimate = partial(
						bin_mle.EM_opt,
						fixpoint=fixpoint,
						update=partial(bin_mle.normalized_dagostini,
									   M=M,
									   int_control=int_control,
									   tol=params.tol),
						init_gamma=init_gamma,
						int_control=int_control,
						# int_omega=params.background.int_omega
					)
				case _:
					raise ValueError('Optimizer not supported')

			return partial(bin_mle.delta_method_test, params=params)

		case _:
			raise ValueError('Method not supported')
