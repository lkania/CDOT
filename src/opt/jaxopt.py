import jax.numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative, projection_polyhedron
from jaxopt.objective import least_squares
from src.opt.error import squared_ls_error
from jaxopt import OSQP
from jaxopt import CvxpyQP


def nnls(A, b, maxiter=1000, tol=1e-4):
    pg = ProjectedGradient(fun=least_squares,
                           verbose=False,
                           acceleration=True,
                           implicit_diff=True,
                           tol=tol,
                           maxiter=maxiter,
                           jit=True,
                           projection=projection_non_negative)
    pg_sol = pg.run(np.zeros((A.shape[1],)), data=(A, b.reshape(-1, )))
    x = pg_sol.params
    return x, squared_ls_error(A=A, b=b, x=x)


def nnls_with_linear_constraint(A, b, c, maxiter=1000, tol=1e-4):
    n_params = A.shape[1]
    pg = ProjectedGradient(fun=least_squares,
                           verbose=False,
                           acceleration=True,
                           implicit_diff=True,
                           tol=tol,
                           maxiter=maxiter,
                           jit=True,
                           projection=lambda x, hyperparams: projection_polyhedron(x=x,
                                                                                   hyperparams=hyperparams,
                                                                                   check_feasible=False))
    # equality constraint
    A_ = c.reshape(1, -1)
    b_ = np.array([1.0])

    # inequality constraint
    G = -1 * np.eye(n_params)
    h = np.zeros((n_params,))

    pg_sol = pg.run(init_params=np.zeros((n_params,)),
                    data=(A, b.reshape(-1, )),
                    hyperparams_proj=(A_, b_, G, h))
    x = pg_sol.params

    return x, squared_ls_error(A=A, b=b, x=x)


def nnls_with_linear_constraint_QP(A, b, c, maxiter=6000, tol=1e-4):
    n_params = A.shape[1]

    # equality constraint
    A_ = c.reshape(1, -1)
    b_ = np.array([1.0])

    # inequality constraint
    G = -1 * np.eye(n_params)
    h = np.zeros((n_params,))

    Q = A.transpose() @ A
    c_ = (-1 * A.transpose() @ b.reshape(-1, 1)).reshape(-1, )

    # TODO: check BoxOSQP, since OSQP gets transformed into BoxOSQP

    qp = OSQP(tol=tol,
              # momentum=1,
              # eq_qp_solve='cg+jacobi',
              primal_infeasible_tol=tol,
              dual_infeasible_tol=tol,
              check_primal_dual_infeasability=False,
              maxiter=maxiter,
              jit=True)

    qp_sol = qp.run(params_obj=(Q, c_),
                    params_eq=(A_, b_),
                    params_ineq=(G, h))

    x = qp_sol.params[0]
    return x, squared_ls_error(A=A, b=b, x=x)
