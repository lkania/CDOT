import jax.random as random
import jax.numpy as np


def add_signal(X, params):
    lambda_ = params.lambda_star
    mu = params.mu_star
    sigma = params.sigma_star
    key = params.key

    if params.no_signal:
        params.X = X
    else:
        n = X.shape[0]
        n_signal = np.int32(n * lambda_)
        signal = mu + sigma * random.normal(key, shape=(n_signal,))
        X_with_signal = np.concatenate((X, signal))
        params.X = X_with_signal

    params.lower = mu - params.std_signal_region * sigma
    params.upper = mu + params.std_signal_region * sigma
