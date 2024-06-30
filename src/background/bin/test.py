from functools import partial

from jax import numpy as np, jit, grad
from jax.scipy.stats.norm import cdf
from jaxopt import ProjectedGradient, GradientDescent
from jaxopt.projection import projection_polyhedron

from src import bin
from src.normalize import normalize, safe_ratio


######################################################################
# local libraries
######################################################################


@jit
def statistic(props, gamma, int_control):
	F_over_control = np.sum(props)
	B_over_control = np.dot(gamma.reshape(-1),
							int_control.reshape(-1))

	lambda_hat = 1 - F_over_control / B_over_control

	stat = B_over_control - F_over_control
	stat = stat / np.sqrt(F_over_control * (1 - F_over_control))
	return stat, lambda_hat


@partial(jit, static_argnames=['tol'])
def dagostini(gamma0, props, M, int_control, tol):
	pred0 = (M @ gamma0.reshape(-1, 1)).reshape(-1)
	props = props.reshape(-1)

	ratio = safe_ratio(num=props, den=pred0, tol=tol)

	ak = M.transpose() @ ratio.reshape(-1, 1)
	ak = ak.reshape(-1)
	gamma = gamma0.reshape(-1) * ak / int_control.reshape(-1)
	return gamma.reshape(-1)


@partial(jit, static_argnames=['tol'])
def normalized_dagostini(gamma0,
						 props,
						 M, int_control, int_omega,
						 tol):
	gamma = dagostini(gamma0=gamma0,
					  props=props,
					  M=M,
					  int_control=int_control,
					  tol=tol)
	return normalize(gamma=gamma, int_omega=int_omega).reshape(-1)


@partial(jit, static_argnames=['tol', 'update', 'fixpoint'])
def EM_opt(props,

		   int_control,
		   int_omega,

		   update,
		   fixpoint,

		   init_gamma):
	sol = fixpoint(fixed_point_fun=update).run(
		# we initialize gamma so that int_Omega B_gamma(x) = 1
		# the fixed point method cannot be differentiated w.r.t
		# the init parameters
		init_gamma.reshape(-1),
		# the fixed point method can be differentiated w.r.t
		# the following auxiliary parameters
		props)

	gamma = sol[0].reshape(-1)

	gamma = normalize(gamma=gamma, int_omega=int_omega)

	stat, lambda_hat = statistic(props=props,
								 gamma=gamma,
								 int_control=int_control)

	aux = dict()
	aux['gamma_hat'] = gamma
	aux['lambda_hat'] = lambda_hat
	aux['stat'] = stat

	return stat, aux


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


def pvalue(zscore):
	return 1 - cdf(zscore, loc=0, scale=1)


@partial(jit, static_argnames=['params'])
def delta_method_test(params, X, mask):
	n = np.sum(mask)

	X_masked = X * mask + (-1) * (1 - mask)

	counts = bin.counts(X=X_masked, from_=params.from_, to_=params.to_)[0]
	counts = counts.reshape(-1)
	props = counts / n

	# delta method for functional based on binning
	grad_op = grad(fun=params.estimate, argnums=0, has_aux=True)
	jac, aux = grad_op(props)

	D_hat = - np.outer(props, props)
	mask = 1 - np.eye(D_hat.shape[0], dtype=props.dtype)
	D_hat = D_hat * mask + np.diag(props * (1 - props))
	t2_hat = np.sum(
		(jac.reshape(1, -1) @ D_hat).reshape(-1) * jac.reshape(-1))
	t_hat = np.sqrt(t2_hat)

	lambda_hat = aux['lambda_hat']
	gamma_hat = aux['gamma_hat']
	zscore = aux['stat']
	pvalue_ = pvalue(np.sqrt(n) * zscore / t_hat)

	#######################################################
	# Store results
	#######################################################
	method = dict()

	method['lambda_hat'] = lambda_hat
	method['gamma_hat'] = gamma_hat
	method['stat'] = pvalue_

	return method
