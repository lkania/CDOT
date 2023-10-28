import jax.numpy as np
from jax import jit


# NOTE:
# 1. the gradient of lstsq aren't the best, it seems like
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
