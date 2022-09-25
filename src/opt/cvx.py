import cvxpy as cp
from cvxpylayers.jax import CvxpyLayer
import jax.numpy as np


# It is incompatible with jacrev
def nnls(A, b):
    # optimization
    l, K = A.shape  # l x K
    b_ = cp.Parameter((l, 1))
    A_ = cp.Parameter((l, K))
    gamma = cp.Variable((K, 1))
    constraints = [gamma >= 0]
    objective = cp.Minimize(cp.sum_squares(A_ @ gamma - b_))
    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem,
                            parameters=[b_, A_],
                            variables=[gamma])

    gamma_ = cvxpylayer(b.reshape(-1, 1), A)[0]
    gamma_ = np.maximum(gamma_, 0)
    return gamma_.reshape(-1, 1)


def ls(A, b):
    # optimization
    l, K = A.shape  # l x K
    A_ = cp.Parameter((l, K))
    b_ = cp.Parameter((l, 1))
    gamma = cp.Variable((K, 1))
    objective = cp.Minimize(cp.sum_squares(A_ @ gamma - b_))
    problem = cp.Problem(objective)
    cvxpylayer = CvxpyLayer(problem,
                            parameters=[b_, A_],
                            variables=[gamma])

    return (cvxpylayer(b.reshape(-1, 1), A)[0]).reshape(-1, 1)
