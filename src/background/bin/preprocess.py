import jax.numpy as np
import src.bin as bin_
from src.background.bin.delta import influence
from jax import jit
from functools import partial
from src.transform import transform


@partial(jit, static_argnames=['basis', 'k'])
def density(gamma, X, k, basis):
    density_ = basis.evaluate(k=k, X=X) @ gamma.reshape(-1, 1)
    return density_.reshape(-1)


@partial(jit, static_argnames=['tilt_density', 'k', 'basis'])
def estimate_background_from_gamma(gamma, tilt_density, X, k, basis):
    background_hat = tilt_density(density=partial(density, k=k, basis=basis, gamma=gamma), X=X)
    return background_hat


def preprocess(params, method):
    assert params.k <= np.int32(params.bins / 2) * 2

    # TODO: We ignore the dependency of tilt_density on min(X),
    #  it can be easily fixed by choosing a reasonable constant
    trans, tilt_density = transform(base=np.min(method.X), c=params.c)
    tX = trans(method.X)
    tlower = trans(params.lower)
    tupper = trans(params.upper)

    # TODO: We ignore the randomness of the equal-counts binning,
    #  it can be fixed by using a fixed binning
    from_, to_ = bin_.bin(X=tX, lower=tlower, upper=tupper, sections=np.int32(params.bins / 2))
    empirical_probabilities, indicators = bin_.proportions(X=tX, from_=from_, to_=to_)
    int_omega = params.basis.int_omega(k=params.k)
    M = params.basis.integrate(params.k, from_, to_)  # n_bins x n_parameters
    int_control = np.sum(M, axis=0).reshape(-1, 1)

    method.background.bins = params.bins
    method.background.from_ = from_
    method.background.to_ = to_

    method.background.int_omega = int_omega
    method.background.M = M
    method.background.int_control = int_control

    # indicators is a n_props x n_obs matrix that indicates to which bin every observation belongs
    method.background.influence = partial(influence,
                                          empirical_probabilities=empirical_probabilities,
                                          indicators=indicators)
    method.background.estimate_background_from_gamma = partial(estimate_background_from_gamma,
                                                               tilt_density=tilt_density,
                                                               k=params.k,
                                                               basis=params.basis)
