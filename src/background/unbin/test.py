from jax import numpy as np, jit, grad
from functools import partial

from jax.scipy.stats.norm import cdf, ppf as icdf
from jaxopt.projection import projection_polyhedron, projection_simplex
from jaxopt import ProjectedGradient, GradientDescent, LBFGS, LBFGSB
from jax import grad, hessian, numpy as np, vmap, jit

from src.basis import bernstein
from src.normalize import normalize, safe_ratio


@partial(jit, static_argnames=['tol'])
def dagostini(gamma0,
			  eval_, mask, in_control_region,
			  int_control,
			  tol):
	# eval_ dim are n_obs x n_params
	background_per_obs = eval_ @ gamma0.reshape(-1, 1)
	background_per_obs = background_per_obs.reshape(-1)
	ratio = safe_ratio(num=in_control_region * mask,
					   den=background_per_obs,
					   tol=tol)
	ak = eval_.transpose() @ ratio.reshape(-1, 1)  # n_params x 1
	ak = ak.reshape(-1)
	gamma = gamma0 * (ak / int_control.reshape(-1))

	return gamma


@partial(jit, static_argnames=['tol'])
def normalized_dagostini(gamma0,
						 eval_, mask, in_control_region,
						 int_control, int_omega,
						 tol):
	gamma = dagostini(gamma0=gamma0,
					  eval_=eval_,
					  mask=mask,
					  in_control_region=in_control_region,
					  int_control=int_control,
					  tol=tol)
	return normalize(gamma=gamma, int_omega=int_omega).reshape(-1)


@partial(jit, static_argnames=['update', 'fixpoint'])
def EM_opt(mask,
		   n,
		   n_control,
		   n_signal,
		   in_control_region,
		   eval_,

		   int_control,
		   int_omega,

		   init_gamma,
		   update,
		   fixpoint):
	sol = fixpoint(fixed_point_fun=update).run(
		# we initialize gamma so that int_Omega B_gamma(x) = 1
		# the fixed point method cannot be differentiated w.r.t
		# the init parameters
		init_gamma.reshape(-1),
		# the fixed point method can be differentiated w.r.t
		# the following auxiliary parameters
		eval_, mask, in_control_region)

	gamma = sol[0]

	gamma_hat = normalize(gamma=gamma, int_omega=int_omega)

	lambda_hat = 1 - (n_control / n) / np.dot(gamma_hat, int_control)

	return lambda_hat, gamma_hat


@partial(jit, static_argnames=['tol'])
def loss_per_obs(vars,
				 eval_,
				 mask,
				 in_control_region,

				 int_signal,

				 tol):
	vars = vars.reshape(-1)
	lambda_ = vars[-1]
	gamma = vars[:-1]

	background_over_signal = np.dot(gamma, int_signal.reshape(-1))
	background_per_obs = eval_ @ gamma.reshape(-1, 1)
	background_per_obs = background_per_obs.reshape(-1)

	back = (1 - lambda_) * background_per_obs
	back = np.where(back <= tol, tol, back)
	t1 = np.log(back) * in_control_region

	signal = (1 - lambda_) * background_over_signal + lambda_
	signal = np.where(signal <= tol, tol, signal)
	t2 = np.log(signal) * (1 - in_control_region)

	return np.squeeze((t1 + t2) * mask)


# The following function is equal to
# np.sum(loss_per_obs(vars=vars,
# 					x=X,
# 					mask=mask,
# 					int_signal=int_signal,
# 					in_control_region=in_control_region,
# 					tol=tol,
# 					k=k))
def loss_(gamma, lambda_,
		  eval_, in_control_region, mask, n_signal,
		  int_signal,
		  tol):
	background_over_signal = np.dot(gamma.reshape(-1), int_signal.reshape(-1))
	background_per_obs = eval_ @ gamma.reshape(-1, 1)
	background_per_obs = background_per_obs.reshape(-1)

	back = (1 - lambda_) * background_per_obs
	back = np.where(back <= tol, tol, back)
	t1 = np.sum(np.log(back) * in_control_region * mask)

	signal = (1 - lambda_) * background_over_signal + lambda_
	signal = np.where(signal <= tol, tol, signal)
	t2 = n_signal * np.log(signal)

	return (-1) * (t1 + t2)


def loss(vars, data, tol):
	eval_, mask, int_signal, n_signal, in_control_region = data

	vars = vars.reshape(-1)
	lambda_ = vars[-1]
	gamma = vars[:-1]

	return loss_(gamma=gamma,
				 lambda_=lambda_,
				 eval_=eval_,
				 in_control_region=in_control_region,
				 mask=mask,
				 n_signal=n_signal,
				 int_signal=int_signal,
				 tol=tol)


def loss_with_opt_lambda(gamma,
						 eval_, mask, n_signal, n_control, n, in_control_region,
						 int_signal, int_control,
						 tol):
	gamma = gamma.reshape(-1)

	background_over_control = np.dot(gamma.reshape(-1), int_control.reshape(-1))
	lambda_ = 1 - (n_control / n) / background_over_control

	return loss_(gamma=gamma,
				 lambda_=lambda_,
				 eval_=eval_,
				 in_control_region=in_control_region,
				 mask=mask,
				 n_signal=n_signal,
				 int_signal=int_signal,
				 tol=tol)


@partial(jit, static_argnames=['tol', 'maxiter', 'projection'])
def constrained_opt(mask,
					n,
					n_signal,
					n_control,
					in_control_region,
					eval_,
					tol,
					maxiter,
					projection,
					init_lambda,
					init_gamma):
	init_lambda = np.array(init_lambda)
	init_params = np.concatenate((init_gamma.reshape(-1),
								  init_lambda.reshape(-1))).reshape(-1, 1)

	pg = ProjectedGradient(
		fun=partial(loss, tol=tol),
		verbose=True,
		acceleration=False,
		implicit_diff=False,
		tol=tol,
		maxiter=maxiter,
		jit=True,
		projection=projection)

	pg_sol = pg.run(
		init_params=init_params,
		eval_=eval_,
		mas=mask,
		n=n,
		n_signal=n_signal,
		n_control=n_control,
		in_control_region=in_control_region
	)
	x = pg_sol.params

	x = x.reshape(-1)
	lambda_ = x[-1]
	gamma = x[:-1].reshape(-1)

	return lambda_, gamma


@partial(jit, static_argnames=['tol', 'maxiter'])
def constrained_opt_with_opt_lambda(
		mask,
		n,
		n_signal,
		n_control,
		in_control_region,
		eval_,
		projection,
		tol,
		maxiter,
		init_gamma):
	pg = ProjectedGradient(
		fun=partial(loss_with_opt_lambda, tol=tol),
		verbose=True,
		acceleration=False,
		implicit_diff=False,
		tol=tol,
		maxiter=maxiter,
		jit=True,
		projection=projection)
	pg_sol = pg.run(
		init_params=init_gamma.reshape(-1),
		eval_=eval_,
		mas=mask,
		n=n,
		n_signal=n_signal,
		n_control=n_control,
		in_control_region=in_control_region
	)
	x = pg_sol.params

	x = x.reshape(-1)
	lambda_ = x[-1]
	gamma = x[:-1].reshape(-1)

	return lambda_, gamma


def unconstrained_opt(
		mask,
		n_signal,
		int_signal,
		int_omega,
		in_control_region,
		eval_,
		stepsize,
		tol,
		maxiter,
		init_lambda,
		init_gamma,
		lambda_lowerbound,
		lambda_upperbound,
		gamma_lowerbound):
	init_lambda = np.array(init_lambda)
	init_params = np.concatenate((init_gamma.reshape(-1),
								  init_lambda.reshape(-1))).reshape(-1, 1)

	n_params = init_gamma.reshape(-1).shape[0]
	lower_bounds = np.zeros((n_params + 1,)) + gamma_lowerbound
	lower_bounds = lower_bounds.at[-1].set(lambda_lowerbound)
	lower_bounds = lower_bounds.reshape(-1)

	upper_bounds = np.ones((n_params + 1,)) * np.inf
	upper_bounds = upper_bounds.at[-1].set(lambda_upperbound)
	upper_bounds = upper_bounds.reshape(-1)

	bounds = (lower_bounds, upper_bounds)

	pg = LBFGSB(
		fun=partial(loss, tol=tol),
		verbose=True,
		# acceleration=False,
		implicit_diff=False,
		tol=tol,
		maxiter=maxiter,
		jit=True)
	pg_sol = pg.run(
		bounds=bounds,
		init_params=init_params.reshape(-1),
		data=(eval_, mask, int_signal, n_signal, in_control_region))
	x = pg_sol.params

	x = x.reshape(-1)
	lambda_ = x[-1]
	gamma = x[:-1].reshape(-1)

	return lambda_, gamma


def pvalue(zscore):
	return 1 - cdf(zscore, loc=0, scale=1)


@partial(jit, static_argnames=['params'])
def efficient_test(params, X, mask):
	n = np.sum(mask)
	X = X.reshape(-1)
	mask = mask.reshape(-1)

	# indicators has n_bins x n_obs
	in_control_region = np.where(X > params.upper, 1, 0) * mask
	in_control_region += np.where(X < params.lower, 1, 0) * mask

	n_control = np.sum(in_control_region)
	n_signal = n - n_control

	# evaluate basis
	eval_ = bernstein.evaluate(k=params.k, X=X)
	eval_ = np.where(eval_ <= params.tol, 0, eval_)

	# check that the basis is a partition of unity
	# assert np.max(np.abs(np.sum(eval_, axis=1) - 1)) < 1e-7

	print('opt start')

	lambda_hat, gamma_hat = params.estimate(
		mask=mask,
		n=n,
		n_control=n_control,
		n_signal=n_signal,
		in_control_region=in_control_region,
		eval_=eval_)

	hessian_op = hessian(
		fun=partial(loss, data=(eval_,
								mask,
								params.background.int_signal.reshape(-1),
								n_signal,
								in_control_region),
					tol=params.tol),
		argnums=0,
		has_aux=False)

	nu_hat = np.concatenate(
		(gamma_hat.reshape(-1), np.array([lambda_hat]))
	).reshape(-1)

	H = hessian_op(nu_hat) / n

	grad_op = grad(partial(loss_per_obs,
						   int_signal=params.background.int_signal.reshape(-1),
						   tol=params.tol),
				   argnums=0,
				   has_aux=False)
	grad_ops = vmap(grad_op, in_axes=(None, 0, 0, 0))
	grad_per_example = grad_ops(nu_hat, eval_, mask, in_control_region)
	psi = np.linalg.solve(H, np.transpose(grad_per_example))
	covar = psi @ np.transpose(psi) / n
	var = covar[-1, -1]

	zscore = np.sqrt(n) * lambda_hat / np.sqrt(var)
	pvalue_ = pvalue(zscore)
	method = dict()

	method['lambda_hat'] = lambda_hat
	method['gamma_hat'] = gamma_hat
	method['stat'] = pvalue_

	return method
