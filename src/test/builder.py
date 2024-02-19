from jax import numpy as np, random, jit, jacrev, grad
from jaxopt import AndersonAcceleration, FixedPointIteration

#######################################################
# Utilities
#######################################################
from functools import partial
from src.dotdic import DotDic
#######################################################
# Basis
#######################################################
from src.basis import bernstein

#######################################################
# background methods
#######################################################
from src.background.bin import mle as bin_mle

#######################################################
# signals
#######################################################
from jax.scipy.stats.norm import pdf as dnorm  # Gaussian/normal signal


def build(args):
	#######################################################
	# init sub-dictionaries
	#######################################################
	params = DotDic()
	params.background = DotDic()

	#######################################################
	# randomness
	#######################################################
	params.seed = int(args.seed)
	# params.key = random.PRNGKey(seed=params.seed)

	#######################################################
	# Background estimation parameters
	#######################################################
	# high impact on jacobian computation for bin methods
	params.bins = int(args.bins)
	# high impact on jacobian computation for non-bin methods
	params.k = int(args.k) if args.k is not None else None

	assert (params.k is None and args.ks is not None) or (
			params.bins >= (params.k + 1) and args.ks is None)

	params.ks = None
	if args.ks is not None:
		assert len(args.ks) >= 2
		params.ks = args.ks
		params.bins_selection = int(
			np.ceil(params.bins * (args.bins_selection / 100) / 2))

		assert (params.bins >= (max(params.ks) + 1))
		assert (params.bins_selection >= 0 and params.bins_selection <= 100)

	#######################################################
	# background transformation
	#######################################################
	params.trans = args.trans
	params.tilt_density = args.tilt_density

	#######################################################
	# Basis
	#######################################################
	params.basis = bernstein

	#######################################################
	# signal region
	#######################################################
	params.lower = args.lower
	params.upper = args.upper
	params.tlower = params.trans(X=params.lower)
	params.tupper = params.trans(X=params.upper)

	#######################################################
	# allow 64 bits
	#######################################################
	params.dtype = np.float64

	#######################################################
	# numerical methods
	#######################################################
	params.tol = args.tol  # convergence criterion of iterative methods
	params.maxiter = args.maxiter

	# params.rcond = 1e-6  # thresholding for singular values

	#######################################################
	# optimizer used in background estimation
	#######################################################

	def fixpoint():
		params.optimizer += '_{0}'.format(args.fixpoint)
		match args.fixpoint:
			case 'normal':
				params.fixpoint = partial(FixedPointIteration,
										  verbose=False,
										  jit=True,
										  implicit_diff=True,
										  tol=params.tol,
										  maxiter=params.maxiter)
			case 'anderson':
				params.fixpoint = partial(AndersonAcceleration,
										  beta=1,
										  history_size=5,
										  mixing_frequency=1,
										  verbose=False,
										  jit=True,
										  implicit_diff=True,
										  tol=params.tol,
										  maxiter=params.maxiter)
			case _:
				raise ValueError('Fixed point solver not supported')

	params.method = args.method
	params.optimizer = args.optimizer
	match params.method:
		case 'bin_mle':
			params.background.fit = bin_mle.fit
			match args.optimizer:
				case 'dagostini':
					fixpoint()
					params.background.optimizer = partial(
						bin_mle.poisson_opt,
						fixpoint=params.fixpoint,
						_delta=bin_mle.dagostini,
						tol=params.tol,
						dtype=params.dtype)
				case 'normalized_dagostini':
					fixpoint()
					params.background.optimizer = partial(
						bin_mle.poisson_opt,
						fixpoint=params.fixpoint,
						_delta=bin_mle.normalized_dagostini,
						tol=params.tol,
						dtype=params.dtype)
				case 'multinomial':
					params.background.optimizer = partial(
						bin_mle.multinomial_opt,
						tol=params.tol,
						maxiter=params.maxiter,
						dtype=params.dtype)
				case 'mom':
					params.background.optimizer = partial(
						bin_mle.mom_opt,
						tol=params.tol,
						maxiter=params.maxiter,
						dtype=params.dtype)
				case _:
					raise ValueError('Optimizer not supported')

	params.grad_op = grad

	return params
