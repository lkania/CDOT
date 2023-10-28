# Let func: R^{n_props} -> R^{p}
# influence returns R^{p} x n_observations
def influence(func, empirical_probabilities, indicators, grad):
    n_probs = empirical_probabilities.shape[0]
    assert indicators.shape[0] == n_probs
    empirical_probabilities = empirical_probabilities.reshape(-1)
    grad_op = grad(fun=func, argnums=0, has_aux=True)
    jac, aux = grad_op(empirical_probabilities)
    influence_ = jac.reshape(-1, n_probs) @ indicators
    return influence_, aux
