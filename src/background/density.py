from jax import jit
from functools import partial


@partial(jit, static_argnames=['basis', 'k'])
def density(gamma, X, k, basis):
    density_ = basis.evaluate(k=k, X=X) @ gamma.reshape(-1, 1)
    return density_.reshape(-1)


@partial(jit, static_argnames=['tilt_density', 'k', 'basis'])
def background(gamma, tilt_density, X, k, basis):
    background_hat = tilt_density(density=partial(density, k=k, basis=basis, gamma=gamma), X=X)
    return background_hat
