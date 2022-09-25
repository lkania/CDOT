from src.basis import bernstein as basis
import jax.numpy as np


def int_omega(k):
    return basis.integrate(k, np.array([0]), np.array([1]))