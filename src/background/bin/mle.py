from src.normalize import normalize, threshold_non_neg

import jax
from jax import jit, numpy as np, vmap
# from src.opt.jaxopt import normalized_nnls_with_linear_constraint
from jaxopt.projection import projection_polyhedron, projection_box_section
from jaxopt import ProjectedGradient, GradientDescent
from src.background.bin.delta import influence, t2_hat
from functools import partial


#########################################################################
# Update rules for poisson nll
#########################################################################

@jit
def _lambda_hat(props, gamma, int_control):
	F_over_control = np.sum(props)
	B_over_control = np.dot(gamma.reshape(-1),
							int_control.reshape(-1))
	return 1 - F_over_control / B_over_control


@partial(jit, static_argnames=['tol'])
def dagostini(gamma0, props, M, int_control, int_omega, tol):
	pred0 = (M @ gamma0.reshape(-1, 1)).reshape(-1)
	props = props.reshape(-1)

	# In the following, we implement this decision tree
	# 	if props <= tol:
	# 		return 0.0
	# 	elif np.abs(props - pred0) <= tol:
	# 		return 1.0
	# 	elif pred0 <= tol:
	# 		return props / tol
	# 	return props / pred0
	c1 = props <= tol
	c2 = np.abs(props - pred0) <= tol
	c3 = pred0 <= tol
	pred = np.where(c1, 0.0, props)
	pred = np.where(np.logical_and(np.logical_not(c1), c2),
					1.0,
					pred)
	c1_or_c2 = np.logical_or(c1, c2)
	pred = np.where(np.logical_and(np.logical_not(c1_or_c2), c3),
					props / tol,
					pred)
	c1_or_c2_or_c3 = np.logical_or(c1_or_c2, c3)
	safe_denominator = np.where(np.logical_not(c1_or_c2_or_c3),
								pred0,
								1.0)
	pred = np.where(np.logical_not(c1_or_c2_or_c3),
					props / safe_denominator,
					pred)

	gamma = (M.transpose() @ pred.reshape(-1, 1)) / int_control.reshape(-1, 1)

	return gamma.reshape(-1)


@partial(jit, static_argnames=['tol'])
def normalized_dagostini(gamma0, props, M, int_control, int_omega, tol):
	return normalize(gamma=dagostini(gamma0=gamma0,
									 props=props,
									 M=M,
									 int_control=int_control,
									 int_omega=int_omega,
									 tol=tol),
					 int_omega=int_omega)


@partial(jit, static_argnames=['_delta'])
def _update(gamma0, props, M, int_control, int_omega, _delta):
	return gamma0 * _delta(gamma0=gamma0, props=props, M=M,
						   int_control=int_control, int_omega=int_omega)


@partial(jit, static_argnames=['tol', '_delta', 'fixpoint'])
def poisson_opt(props,
				M,
				int_control,
				int_omega,
				tol,
				_delta,
				fixpoint,
				init_lambda,
				init_gamma):
	sol = fixpoint(fixed_point_fun=partial(_update, _delta=_delta)).run(
		# we initialize gamma so that int_Omega B_gamma(x) = 1
		# the fixed point method cannot be differentiated w.r.t
		# the init parameters
		init_gamma.reshape(-1),
		# the fixed point method can be differentiated w.r.t
		# the following auxiliary parameters
		props, M, int_control, int_omega)

	gamma = sol[0]
	
	gamma = normalize(gamma=gamma, int_omega=int_omega)

	lambda_ = _lambda_hat(props=props,
						  gamma=gamma,
						  int_control=int_control)

	aux = dict()
	aux['gamma_hat'] = gamma
	aux['lambda_hat'] = lambda_

	return lambda_, aux


@partial(jit, static_argnames=['tol'])
def loss(gamma, data, tol):
	M, props, int_control, int_omega = data
	props = props.reshape(-1)
	gamma = gamma.reshape(-1)
	lambda_ = gamma[-1]
	gamma = gamma[:-1]

	prop_control = np.sum(props)
	background_over_signal = 1 - np.dot(gamma, int_control.reshape(-1))
	background_over_control_bins = (M @ gamma.reshape(-1, 1)).reshape(-1)
	t1 = prop_control * np.log(1 - lambda_)
	pred = np.where(background_over_control_bins <= tol,
					tol,
					background_over_control_bins)
	pred = np.log(pred)
	pred = np.where(props <= tol, 0.0, props * pred)
	t2 = np.sum(pred)
	mix = (1 - lambda_) * background_over_signal + lambda_
	mix = np.where(mix <= tol, tol, mix)
	t3 = (1 - prop_control) * np.log(mix)
	loss_ = (-1) * (t1 + t2 + t3)

	return loss_


# With the current optimization
# lambda is a boundary point
# this leads to upper bias in its estimation
# Empirically, I also see bias when using
# negative gammas
@partial(jit, static_argnames=['tol', 'maxiter'])
def multinomial_opt(props,
					M,
					int_control,
					int_omega,
					tol,
					maxiter,
					init_lambda,
					init_gamma):
	zero = np.array([0])
	lambda_upperbound = np.array([0.9])
	lambda_lowerbound = 0
	gamma_lowerbound = 0

	int_omega = int_omega.reshape(-1)
	n_params = M.shape[1]

	# Equality constraint
	# force the basis to ingreate to 1 over the omega domain
	A_ = (np.concatenate((int_omega, zero)).reshape(-1)).reshape(1, -1)
	b_ = np.array([1.0])

	# If using projection_polyhedron
	# then use the following inequality constraints
	G = -1 * np.eye(n_params + 2, n_params + 1)
	G = G.at[-1].set(np.zeros(n_params + 1, ).at[-1].set(1))
	lower_bounds = np.zeros((n_params + 1,)) + gamma_lowerbound
	lower_bounds = lower_bounds.at[-1].set(lambda_lowerbound)
	h = np.concatenate((lower_bounds, lambda_upperbound))

	init_lambda = np.array(init_lambda)  # np.array([init_lambda])
	init_params = np.concatenate((init_gamma.reshape(-1),
								  init_lambda.reshape(-1))).reshape(-1, 1)

	pg = ProjectedGradient(
		fun=partial(loss, tol=tol),
		verbose=False,
		acceleration=True,
		implicit_diff=False,
		tol=tol,
		maxiter=maxiter,
		jit=True,
		projection=partial(projection_polyhedron,
						   check_feasible=False))
	pg_sol = pg.run(
		init_params=init_params.reshape(-1),
		data=(M, props, int_control, int_omega),
		hyperparams_proj=(A_, b_, G, h))
	x = pg_sol.params

	x = x.reshape(-1)
	lambda_ = x[-1]
	gamma = x[:-1].reshape(-1, 1)

	aux = dict()
	aux['gamma_hat'] = gamma
	aux['lambda_hat'] = lambda_

	return lambda_, aux


@partial(jit, static_argnames=['tol', 'maxiter'])
def multinomial_opt2(props,
					 M,
					 int_control,
					 int_omega,
					 tol,
					 maxiter,
					 init_lambda,
					 init_gamma):
	int_omega = int_omega.reshape(-1)
	init_lambda = np.array(init_lambda)
	init_params = np.concatenate((init_gamma.reshape(-1),
								  init_lambda.reshape(-1)))

	pg = GradientDescent(
		fun=partial(loss, tol=tol),
		stepsize=tol,
		verbose=False,
		acceleration=False,
		implicit_diff=True,
		tol=tol,
		maxiter=maxiter,
		jit=True)

	pg_sol = pg.run(
		init_params=init_params.reshape(-1),
		data=(M, props, int_control, int_omega))
	x = pg_sol.params

	x = x.reshape(-1)
	lambda_ = x[-1]
	gamma = x[:-1].reshape(-1, 1)

	# new lambda_ computation to avoid boundary issue
	# lambda_ = _lambda_hat(props=props,
	# 					  gamma=gamma,
	# 					  int_control=int_control)

	aux = dict()
	aux['gamma_hat'] = gamma
	aux['lambda_hat'] = lambda_

	return lambda_, aux


def preprocess(params):
	return 0


# params.background.loss = lambda props, gamma, lambda_: loss(
# 	gamma=np.concatenate((gamma.reshape(-1), np.array([lambda_]))),
# 	data=(M, props, int_control, int_omega),
# 	tol=params.tol)

# params.background.estimate_lambda2 = partial(
# 	params.background.optimizer2,
# 	M=params.background.M,
# 	int_control=params.background.int_control,
# 	int_omega=params.background.int_omega)

# params.background.estimate_gamma = partial(
# 	params.background.optimizer,
# 	M=params.background.M,
# 	int_control=params.background.int_control,
# 	int_omega=params.background.int_omega)
#
# params.background.estimate_lambda = partial(
# 	estimate_lambda,
# 	compute_gamma=params.background.estimate_gamma,
# 	int_control=params.background.int_control)


# indicators is a n_props x n_obs matrix that indicates
# to which bin every observation belongs
# params.background.influence = partial(
# 	influence,
# 	grad=params.grad_op)

# params.background.t2_hat = partial(
# 	t2_hat,
# 	grad=params.grad_op)


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
