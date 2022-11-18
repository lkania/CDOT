import jax.numpy as np
import src.bin as bin_
from src.background.bin.delta import influence
from functools import partial
from src.transform import transform
from src.background.density import background


def preprocess(params, method):
    assert params.k <= np.int32(params.bins / 2) * 2

    trans, tilt_density, _ = transform(a=params.a, b=params.b, rate=params.rate)
    tX = trans(X=method.X)
    tlower = trans(X=params.lower)
    tupper = trans(X=params.upper)

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

    # indicators is a n_props x n_obs matrix that indicates
    # to which bin every observation belongs
    method.background.influence = partial(
        influence,
        empirical_probabilities=empirical_probabilities,
        indicators=indicators)
    method.background.estimate_background_from_gamma = partial(
        background,
        tilt_density=tilt_density,
        k=params.k,
        basis=params.basis)
