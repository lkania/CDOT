from functools import partial

from jax import numpy as np, jit, grad
from jax.scipy.stats.norm import cdf
from jaxopt import ProjectedGradient, GradientDescent
from jaxopt.projection import projection_polyhedron

from src import bin, normalize


######################################################################
# local libraries
######################################################################


@jit
def statistic(props, gamma, int_control):
	Fn_over_control = np.sum(props)
	Background_over_control = np.dot(gamma.reshape(-1), int_control.reshape(-1))

	lambda_hat = 1 - Fn_over_control / Background_over_control

	# stat = Background_over_control - Fn_over_control
	# stat = stat / np.sqrt(Fn_over_control * (1 - Fn_over_control))
	return lambda_hat, lambda_hat


@partial(jit, static_argnames=['tol'])
def dagostini(gamma0, props, M, int_control, tol):
	pred0 = (M @ gamma0.reshape(-1, 1)).reshape(-1)
	props = props.reshape(-1)

	ratio = normalize.safe_ratio(num=props, den=pred0, tol=tol)

	ak = M.transpose() @ ratio.reshape(-1, 1)
	ak = ak.reshape(-1)
	gamma = gamma0.reshape(-1) * ak / int_control.reshape(-1)
	return gamma.reshape(-1)


@partial(jit, static_argnames=['tol'])
def normalized_dagostini(gamma0,
						 props,
						 M, int_control,
						 tol):
	gamma = dagostini(gamma0=gamma0,
					  props=props,
					  M=M,
					  int_control=int_control,
					  tol=tol)

	# assumes normalized basis, i.e. int_Omega phi_k(x) =1
	return (gamma / np.sum(gamma)).reshape(-1)


@partial(jit, static_argnames=['tol', 'update', 'fixpoint'])
def EM_opt(props,

		   int_control,

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

	# assumes normalized basis, i.e. int_Omega phi_k(x) =1
	gamma = (gamma / np.sum(gamma)).reshape(-1)

	stat, lambda_hat = statistic(props=props,
								 gamma=gamma,
								 int_control=int_control)

	aux = dict()
	aux['gamma_hat'] = gamma
	aux['lambda_hat'] = lambda_hat
	aux['stat'] = stat

	return stat, aux


@partial(jit, static_argnames=['tol'])
def loss(gamma, props, M, int_control, tol):
	props = props.reshape(-1)
	gamma = gamma.reshape(-1)

	prop_control = np.sum(props)
	background_over_control = np.squeeze(np.dot(gamma, int_control.reshape(-1)))
	background_over_signal = 1 - background_over_control
	background_over_control_bins = (M @ gamma.reshape(-1, 1)).reshape(-1)

	lambda_ = 1 - prop_control / background_over_control

	pred = normalize.safe_log(x=(1 - lambda_) * background_over_control_bins,
							  tol=tol)
	pred = np.where(props <= tol, 0.0, props * pred)
	t1 = np.sum(pred)

	t2 = (1 - prop_control) * normalize.safe_log(
		x=(1 - lambda_) * background_over_signal + lambda_,
		tol=tol)

	return (-1) * (t1 + t2)


@partial(jit, static_argnames=['tol', 'maxiter', 'loss', 'projection'])
def constrained_opt(props,
					int_control,
					tol,
					maxiter,
					init_gamma,
					loss,
					projection):
	pg = ProjectedGradient(
		fun=loss,
		verbose=False,
		acceleration=True,
		implicit_diff=True,
		tol=tol,
		maxiter=maxiter,
		jit=True,
		projection=projection)
	pg_sol = pg.run(
		init_params=init_gamma.reshape(-1),
		props=props)
	gamma = pg_sol.params.reshape(-1)

	stat, lambda_hat = statistic(props=props,
								 gamma=gamma,
								 int_control=int_control)

	aux = dict()
	aux['gamma_hat'] = gamma
	aux['lambda_hat'] = lambda_hat
	aux['stat'] = stat

	return lambda_hat, aux


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

	# compute empirical covariance matrix
	D_hat = - np.outer(props, props)
	mask = 1 - np.eye(D_hat.shape[0], dtype=props.dtype)
	D_hat = D_hat * mask + np.diag(props * (1 - props))

	# compute variance via discrete delta method
	grad_op = grad(fun=params.estimate, argnums=0, has_aux=True)
	jac, aux = grad_op(props)
	t2_hat = np.sum(
		(jac.reshape(1, -1) @ D_hat).reshape(-1) * jac.reshape(-1))
	t_hat = np.sqrt(t2_hat)

	lambda_hat = aux['lambda_hat']
	gamma_hat = aux['gamma_hat']
	zscore = aux['stat']
	pvalue_ = pvalue(np.sqrt(n) * zscore / t_hat)

	method = dict()
	method['lambda_hat'] = lambda_hat
	method['gamma_hat'] = gamma_hat
	method['stat'] = pvalue_

	return method
