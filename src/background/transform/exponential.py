from jax import numpy as np, jit
from functools import partial


@partial(jit, static_argnames=['rate', 'base'])
def _trans(X, rate, base):
    return np.exp(-rate * (X - base))


@partial(jit, static_argnames=['rate', 'base'])
def trans(X, rate, base, scale):
    return (1 - _trans(X=X, rate=rate, base=base)) / scale


@partial(jit, static_argnames=['rate', 'base'])
def inv_trans(X, rate, base, scale):
    return np.log(1 - X * scale) / (-rate) + base


# we are computing density(trans(X=X,c=c,base=base)) * c * _trans(X=X,c=c,base=base)
# avoiding re-computation
@partial(jit, static_argnames=['density', 'rate', 'base'])
def tilt_density(density, X, rate, base, scale):
    trans_aux = _trans(X=X, rate=rate, base=base)
    trans_ = (1 - trans_aux) / scale
    return density(X=trans_) * rate * trans_aux / scale


def transform(rate, a, b):
    base = a
    scale = 1 if b is None else (1 - _trans(X=b, rate=rate, base=base))
    return partial(trans, rate=rate, base=base, scale=scale), \
           partial(tilt_density, rate=rate, base=base, scale=scale), \
           partial(inv_trans, rate=rate, base=base, scale=scale)
