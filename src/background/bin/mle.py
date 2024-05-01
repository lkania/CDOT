from src.normalize import normalize, threshold
from src.background.bin.lambda_hat import estimate_lambda

from jax import jit, numpy as np
from functools import partial
from src.opt.jaxopt import normalized_nnls_with_linear_constraint
from jaxopt.projection import projection_polyhedron
from jaxopt import ProjectedGradient
from src.bin import proportions
from src.background.bin.delta import influence, t2_hat
from functools import partial


#########################################################################
# Update rules for poisson nll
#########################################################################

# Note that thresholding small predictions seems to
# severely affect the delta method.
@jit
def dagostini(gamma0, props, M, int_control, int_omega):
	pred0 = M @ gamma0.reshape(-1, 1)
	# If the mean prediction is zero, we leave it at zero
	# we use the double-where trick to handle zero values
	# see: https://github.com/google/jax/issues/5039
	pred = np.where(pred0 != 0.0, pred0, 0.0)
	pred = np.where(pred0 != 0.0, props.reshape(-1, 1) / pred, 0.0)
	gamma = (M.transpose() @ pred) / int_control.reshape(-1, 1)

	return gamma


@jit
def normalized_dagostini(gamma0, props, M, int_control, int_omega):
	return normalize(gamma=dagostini(gamma0=gamma0,
									 props=props,
									 M=M,
									 int_control=int_control,
									 int_omega=int_omega),
					 int_omega=int_omega)


@partial(jit, static_argnames=['_delta'])
def _update(gamma0, props, M, int_control, int_omega, _delta):
	return gamma0 * _delta(gamma0=gamma0, props=props, M=M,
						   int_control=int_control, int_omega=int_omega)


@partial(jit, static_argnames=['tol', '_delta', 'fixpoint'])
def poisson_opt(props, M, int_control, int_omega, tol, _delta, fixpoint):
	sol = fixpoint(fixed_point_fun=partial(_update, _delta=_delta)).run(
		# we initialize gamma so that int_Omega B_gamma(x) = 1
		# the fixed point method cannot be differentiated w.r.t
		# the init parameters
		np.full_like(int_control, 1, dtype=int_control.dtype) / np.sum(
			int_omega),
		# the fixed point method can be differentiated w.r.t
		# the following auxiliary parameters
		props, M, int_control, int_omega)

	gamma = normalize(
		gamma=threshold(sol[0], tol=tol),
		int_omega=int_omega)

	gamma_error = np.max(np.abs(_delta(gamma0=gamma,
									   props=props,
									   M=M,
									   int_control=int_control,
									   int_omega=int_omega) - 1))

	gamma_aux = (gamma_error,
				 poisson_nll(gamma=gamma, data=(M, props, int_control)),
				 multinomial_nll(gamma=gamma, data=(M, props, int_control)))

	return gamma, gamma_aux


# @partial(jit, static_argnames=['dtype', 'tol', 'maxiter', 'dtype'])
def multinomial_opt(props, M, int_control, int_omega,
					tol, maxiter, dtype):
	A = M
	b = props.reshape(-1)
	c = int_omega.reshape(-1)
	n_params = A.shape[1]
	pg = ProjectedGradient(
		fun=multinomial_nll,
		verbose=False,
		acceleration=True,
		implicit_diff=False,
		tol=tol,
		maxiter=maxiter,
		jit=False,
		projection=
		lambda x, hyperparams: projection_polyhedron(
			x=x,
			hyperparams=hyperparams,
			check_feasible=False))
	# equality constraint
	A_ = c.reshape(1, -1)
	b_ = np.array([1.0])

	# inequality constraint
	G = -1 * np.eye(n_params)
	h = np.zeros((n_params,))

	pg_sol = pg.run(
		init_params=np.full_like(c, 1, dtype=dtype) / n_params,
		data=(A, b, c),
		hyperparams_proj=(A_, b_, G, h))
	x = pg_sol.params

	gamma = normalize(threshold(x, tol=tol), int_omega=int_omega)

	gamma_error = 0
	gamma_aux = (gamma_error,
				 poisson_nll(gamma=gamma, data=(M, props, int_control)),
				 multinomial_nll(gamma=gamma, data=(M, props, int_control)))

	return gamma, gamma_aux


# @partial(jit, static_argnames=['dtype', 'tol', 'maxiter', 'dtype'])
def mom_opt(props, M, int_control, int_omega,
			tol, maxiter, dtype):
	gamma, gamma_error = normalized_nnls_with_linear_constraint(
		b=props / np.sum(props),
		A=M,
		c=int_omega,
		maxiter=maxiter,
		tol=tol,
		dtype=dtype)

	gamma_aux = (gamma_error,
				 poisson_nll(gamma=gamma, data=(M, props, int_control)),
				 multinomial_nll(gamma=gamma, data=(M, props, int_control)))

	return gamma, gamma_aux


def preprocess(params, method):
	int_omega = params.basis.int_omega(k=method.k)
	assert not np.isnan(int_omega).any()

	M = params.basis.integrate(method.k,
							   params.from_,
							   params.to_)  # n_bins x n_parameters
	assert not np.isnan(M).any()

	int_control = np.sum(M, axis=0).reshape(-1, 1)
	assert not np.isnan(int_control).any()

	method.background.int_omega = int_omega
	method.background.M = M
	method.background.int_control = int_control

	method.background.estimate_gamma = partial(
		params.background.optimizer,
		M=method.background.M,
		int_control=method.background.int_control,
		int_omega=method.background.int_omega)

	method.background.estimate_lambda = partial(
		estimate_lambda,
		compute_gamma=method.background.estimate_gamma,
		int_control=method.background.int_control)

	# indicators is a n_props x n_obs matrix that indicates
	# to which bin every observation belongs
	method.background.influence = partial(
		influence,
		empirical_probabilities=method.background.empirical_probabilities,
		indicators=method.background.indicators,
		grad=params.grad_op)

	method.background.t2_hat = partial(
		t2_hat,
		empirical_probabilities=method.background.empirical_probabilities,
		grad=params.grad_op)


def fit(params, method):
	empirical_probabilities, indicators = proportions(
		X=method.X,
		from_=params.from_,
		to_=params.to_)
	assert not np.isnan(empirical_probabilities).any()

	method.background.empirical_probabilities = empirical_probabilities
	method.background.indicators = indicators


@jit
def multinomial_nll(gamma, data):
	M, props, int_control = data
	background_over_control = np.dot(gamma.reshape(-1), int_control.reshape(-1))
	background_over_bins = (M @ gamma.reshape(-1, 1)).reshape(-1)
	log_ratio = np.log(background_over_bins) - np.log(background_over_control)
	return (-1) * np.sum(props.reshape(-1) * log_ratio)


@jit
def poisson_nll(gamma, data):
	M, props, int_control = data
	log_preds = np.log((M @ gamma.reshape(-1, 1)).reshape(-1))
	int_over_control = np.dot(gamma.reshape(-1), int_control.reshape(-1))
	return (-1) * (np.sum(props.reshape(-1) * log_preds) - int_over_control)
