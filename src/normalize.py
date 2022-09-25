import jax.numpy as np


def normalize(gamma_and_aux, int_omega):
    gamma, gamma_error = gamma_and_aux
    normalized_gamma = gamma.reshape(-1, 1) / np.dot(gamma.reshape(-1),
                                                     int_omega.reshape(-1))
    return normalized_gamma, gamma_error
