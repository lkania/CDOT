import jax.numpy as np


def add_signal(X, params, method):
    X = X.reshape(-1)

    method.background.X = X
    if params.no_signal:
        method.X = X
    else:
        n = X.shape[0]
        n_signal = np.int32(n * params.lambda_star)
        signal = params.signal.sample(n_signal)
        signal = signal.reshape(-1)
        method.X = np.concatenate((X, signal))
