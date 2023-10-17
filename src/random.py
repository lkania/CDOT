import jax.random as random
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
        X_with_signal = np.concatenate((X, signal))
        method.X = X_with_signal
        method.signal.X = signal

    # # sample splitting
    # if params.sample_split:
    #     idx = random.permutation(key=params.key,
    #                              x=np.arange(method.X.shape[0]),
    #                              independent=True)
    #     idxs = np.array_split(idx, indices_or_sections=2)
    #     method.X = method.X[idxs[0]]
    #     method.X2 = method.X[idxs[1]]
