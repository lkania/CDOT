import jax.numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_polyhedron, projection_non_negative
from jaxopt.objective import least_squares
from src.opt.error import squared_ls_error
from src.normalize import normalize, threshold_non_neg
from jax import jit


# @partial(jit, static_argnames=['maxiter', 'tol', 'dtype'])
def nnls_with_linear_constraint(A, b, c, maxiter, tol, dtype):
	n_params = A.shape[1]
	pg = ProjectedGradient(
		fun=least_squares,
		verbose=False,
		acceleration=True,
		implicit_diff=False,
		tol=tol,
		maxiter=maxiter,
		jit=False,
		projection=lambda x, hyperparams: projection_polyhedron(
			x=x,
			hyperparams=hyperparams,
			check_feasible=False))

	# equality constraint
	A_ = c.reshape(1, -1)
	b_ = np.array([1.0])

	# inequality constraint
	G = -1 * np.eye(n_params)
	h = np.zeros((n_params,))

	pg_sol = pg.run(init_params=np.ones((n_params,)) / np.sum(c),
					data=(A, b.reshape(-1, )),
					hyperparams_proj=(A_, b_, G, h))
	x = pg_sol.params

	x = normalize(threshold_non_neg(x, tol=tol), int_omega=c)

	return x, squared_ls_error(A=A, b=b, x=x)


@jit
def _normalized_ls_objective(x, data):
	A, b = data
	pred = A @ x.reshape(-1, 1)
	return np.sum(np.square(b.reshape(-1, 1) - pred / np.sum(pred)))


# @partial(jit, static_argnames=['maxiter', 'tol', 'dtype'])
def normalized_nnls_with_linear_constraint(A, b, c, maxiter, tol, dtype):
	n_params = A.shape[1]
	pg = ProjectedGradient(
		fun=_normalized_ls_objective,
		verbose=False,
		acceleration=True,
		implicit_diff=False,
		# cannot enable implicit_diff due to custom vjp
		# not implemented for _normalized_ls_objective
		tol=tol,
		maxiter=maxiter,
		jit=False,
		projection=lambda x, hyperparams: projection_polyhedron(
			x=x,
			hyperparams=hyperparams,
			check_feasible=False)
	)
	# equality constraint
	A_ = c.reshape(1, -1)
	b_ = np.array([1.0])

	# inequality constraint
	G = -1 * np.eye(n_params)
	h = np.zeros((n_params,))

	data = (A, b.reshape(-1, ))
	pg_sol = pg.run(init_params=np.ones((n_params,)) / np.sum(c),
					data=data,
					hyperparams_proj=(A_, b_, G, h))
	x = pg_sol.params

	x = normalize(threshold_non_neg(x, tol=tol), int_omega=c)

	return x, _normalized_ls_objective(x=x, data=data)
