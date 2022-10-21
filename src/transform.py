from jax import numpy as np, jit
from functools import partial


@partial(jit, static_argnames=['c'])
def _trans(X, c, base):
    return np.exp(-c * (X - base))


@partial(jit, static_argnames=['c'])
def trans(X, c, base):
    return 1 - _trans(X=X, c=c, base=base)


@partial(jit, static_argnames=['c'])
def inv_trans(X, c, base):
    return np.log(1 - X) / (-c) + base


# we are computing density(trans(X=X,c=c,base=base)) * c * _trans(X=X,c=c,base=base)
# avoiding re-computation
@partial(jit, static_argnames=['density', 'c'])
def tilt_density(density, X, c, base):
    trans_aux = _trans(X=X, c=c, base=base)
    trans_ = 1 - trans_aux
    return density(X=trans_) * c * trans_aux


def transform(base, c):
    return partial(trans, base=base, c=c), \
           partial(tilt_density, c=c, base=base), \
           partial(inv_trans, base=base, c=c)
