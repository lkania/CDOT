import jax.numpy as np
from jax import jit


# import jax.scipy as jsp


# NOTES:
# TODO: 1. the gradient of lstsq aren't the best, it seems like
# they are propagating through SVD
# It's better to use the implicit differentiation approach
# - https://github.com/google/jax/issues/10805

# 2. lstsq is inaccurate when using single float precision
# (i.e. 32 rather than 64 bits)
# - https://github.com/google/jax/issues/11433
# In 64 bits, lstsq matches the precision required by nnls
# (I checked it by hand against scipy.nnls in a few examples)
@jit
def ls(A, b):
    return np.linalg.lstsq(A, b.reshape(-1))[0]

# The function has dedicated algebraic gradients
# https://github.com/google/jax/pull/2220
# The precision is not good
# def ls(A, b):
#     return np.linalg.solve(A.transpose() @ A, A.transpose() @ b.reshape(-1, 1))


# The function has dedicated algebraic gradients
# https://github.com/google/jax/pull/2220
# Can produce nan's due to lack of precision
# def ls(A, b):
#     return jsp.linalg.solve(A.transpose() @ A, A.transpose() @ b.reshape(-1, 1),
#                             assume_a='pos')
