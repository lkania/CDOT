import jax.numpy as np


def transform(X, c=0.003):
    X_min = np.min(X)
    trans_ = lambda x: np.exp(-c * (x - X_min))
    trans = lambda x: 1 - trans_(x)
    tilt_density = lambda density, x: density(trans(x)) * c * trans_(x)

    return trans, tilt_density
