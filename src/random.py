import jax.random as random
import jax.numpy as np


def add_signal(X, params, method):
    X = X.reshape(-1)
    if params.no_signal:
        method.X = X
    else:
        n = X.shape[0]
        n_signal = np.int32(n * params.lambda_star)
        signal = params.mu_star + params.sigma_star * random.normal(params.key, shape=(n_signal,))
        signal = signal.reshape(-1)
        X_with_signal = np.concatenate((X, signal))
        method.X = X_with_signal
        method.signal.X = signal

    method.background.X = X
