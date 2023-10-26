# Let func: R^{n_props} -> R^{p}
# influence returns R^{p} x n_observations
# TODO: originally jacrev was used to support non-scalar func
def influence(func, empirical_probabilities, indicators, grad):
    empirical_probabilities = empirical_probabilities.reshape(-1)
    n_probs = empirical_probabilities.shape[0]
    grad_op = grad(fun=func, argnums=0, has_aux=True)
    jac, aux = grad_op(empirical_probabilities)
    influence_ = jac.reshape(-1, n_probs) @ indicators.reshape(n_probs, -1)
    return influence_, aux
