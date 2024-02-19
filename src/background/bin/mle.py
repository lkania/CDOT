from src.background.bin.preprocess import preprocess
from src.normalize import normalize, threshold
from src.background.bin.lambda_hat import estimate_lambda

from jax import jit, numpy as np
from functools import partial
from src.opt.jaxopt import normalized_nnls_with_linear_constraint
from jaxopt.projection import projection_polyhedron
from jaxopt import ProjectedGradient


#########################################################################
# Update rules for poisson nll
#########################################################################

@jit
def dagostini(gamma0, props, M, int_control, int_omega):
	pred = props.reshape(-1, 1) / (M @ gamma0.reshape(-1, 1))
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


@partial(jit, static_argnames=['dtype', 'tol', 'dtype', '_delta', 'fixpoint'])
def poisson_opt(props, M, int_control, int_omega,
				tol, dtype, _delta, fixpoint):
	sol = fixpoint(fixed_point_fun=partial(_update, _delta=_delta)).run(
		# we initialize gamma so that int_Omega B_gamma(x) = 1
		# the fixed point method cannot be differentiated w.r.t
		# the init parameters
		np.full_like(int_control, 1, dtype=dtype) / np.sum(int_omega),
		# the fixed point method can be differentiated w.r.t
		# the following auxiliary parameters
		props, M, int_control, int_omega)

	gamma = normalize(
		gamma=threshold(sol[0], tol=tol, dtype=dtype),
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

	gamma = normalize(
		threshold(x, tol=tol, dtype=dtype), int_omega=int_omega)

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


def fit(params, method):
	preprocess(params=params, method=method)

	method.background.estimate_gamma = partial(
		params.background.optimizer,
		M=method.background.M,
		int_control=method.background.int_control,
		int_omega=method.background.int_omega)

	method.background.estimate_lambda = partial(
		estimate_lambda,
		compute_gamma=method.background.estimate_gamma,
		int_control=method.background.int_control)


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
