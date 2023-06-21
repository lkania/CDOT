import jax.numpy as np
from jax import jit
from jax.scipy.stats.norm import pdf as dnorm  # Gaussian/normal signal


@jit
def signal(X, mu, sigma2):
    return dnorm(x=X.reshape(-1), loc=mu, scale=np.sqrt(sigma2)).reshape(-1)
