import jax.numpy as np
from src.bin import adaptive_bin, proportions, indicator, _adaptive_bin
from src.background.bin.delta import influence
from functools import partial
from src.background.density import background


def preprocess(params, method):
    assert params.k <= (params.bins + 1)

    method.tX = params.trans(X=method.X)

    # TODO: We ignore the randomness of the equal-counts binning,
    #  it can be fixed by using a fixed binning
    from_, to_ = adaptive_bin(X=method.tX,
                              lower=params.tlower,
                              upper=params.tupper,
                              n_bins=params.bins)

    ####################################################################
    # modifications for model selection
    ####################################################################
    # we reserve some bins for model selection
    # that is, compute gamma on fewer bins
    # and then predict bins around the signal region
    if params.model_selection:
        n_model_selection = params.bins_selection
        s_lower, s_upper = _adaptive_bin(X=method.tX,
                                         lower=params.tlower,
                                         upper=params.tupper,
                                         n_bins=params.bins)

        # choose bins around signal region to check the extrapolation of the model
        _sel_lower = s_lower[-n_model_selection:]
        _sel_upper = s_upper[:n_model_selection]
        _s_lower = s_lower[:-n_model_selection]
        _s_upper = s_upper[n_model_selection:]

        _from_ = np.concatenate(
            (np.array([0]), _s_lower, np.array([_sel_upper[-1]]), _s_upper))
        _to_ = np.concatenate(
            (_s_lower, np.array([_sel_lower[0]]), _s_upper, np.array([1])))

        sel_from_ = np.concatenate(
            (_sel_lower, np.array([params.tupper]), _sel_upper[:-1]))
        sel_to_ = np.concatenate(
            (_sel_lower[1:], np.array([params.tlower]), _sel_upper))

        from_ = _from_
        to_ = _to_

        props_val = proportions(X=method.tX,
                                from_=sel_from_, to_=sel_to_)[0].reshape(-1)
        basis_val = params.basis.integrate(params.k, sel_from_, sel_to_)

    ####################################################################
    # bookeeping
    ####################################################################
    empirical_probabilities, indicators = proportions(X=method.tX,
                                                      from_=from_,
                                                      to_=to_)
    method.background.empirical_probabilities = empirical_probabilities

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
        tilt_density=params.tilt_density,
        k=params.k,
        basis=params.basis)

    # for model selection
    if params.model_selection:
        method.background.validation = lambda compute_gamma: np.sqrt(np.sum(
            np.square(
                (basis_val @ compute_gamma(empirical_probabilities)[0].reshape(
                    -1, 1)).reshape(-1) - props_val)))
