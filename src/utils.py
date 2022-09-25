import jax.numpy as np


# split signal and control regions
def extract_control(X, lower, upper):
    idx_lower = X < lower
    idx_upper = X > upper
    idx_control = np.logical_and(idx_lower,idx_upper)
    idx_control = np.array(idx_control, dtype=np.bool_)

    X_lower = X[idx_lower]
    X_upper = X[idx_upper]
    X_control = X[idx_control]
    X_signal = X[np.logical_not(idx_control)]

    return X_control, X_signal, X_lower, X_upper
