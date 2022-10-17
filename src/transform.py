from jax import numpy as np, jit
from functools import partial


def _trans(X, c, base):
    return np.exp(-c * (X - base))


def trans(X, c, base):
    return 1 - _trans(X=X, c=c, base=base)


# we are computing density(trans(X=X,c=c,base=base)) * c * _trans(X=X,c=c,base=base)
# avoiding re-computation
def tilt_density(density, X, c, base):
    trans_aux = _trans(X=X, c=c, base=base)
    trans_ = 1 - trans_aux
    return density(X=trans_) * c * trans_aux


def transform(base, c):
    return partial(trans, base=base, c=c), partial(tilt_density, c=c, base=base)
