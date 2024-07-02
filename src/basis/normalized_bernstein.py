from jax import jit
from functools import partial
from src.basis import bernstein


@partial(jit, static_argnames=['k'])
def evaluate(k, X):
	return bernstein.evaluate(k, X) * (k + 1)


@partial(jit, static_argnames=['k'])
def integrate(k, a, b):
	return bernstein.integrate(k, a, b) * (k + 1)


@partial(jit, static_argnames=['k'])
def predict(gamma, k, from_, to_):
	return integrate(k=k, a=from_, b=to_) @ gamma.reshape(-1, 1)
