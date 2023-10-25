from src.load import load
from src.random import add_signal
import jax.random as random
import jax.numpy as np
from tqdm import tqdm
from src.dotdic import DotDic
from src.ci.delta import delta_cis
import numpy as onp


def _evaluate(X, params):
    method = DotDic()
    method.name = params.name
    method.k = params.k

    add_signal(X=X, params=params, method=method)

    if params.k is None:
        val_error = onp.zeros(len(params.k_grid))
        for i, k in enumerate(params.k_grid):
            params.k = k
            params.background.fit(params=params, method=method)
            val_error[i] = method.background.validation_error()
        k_star = onp.argmin(val_error)
        params.k = params.k_grid[k_star]

    params.background.fit(params=params, method=method)

    params.signal.fit(params=params, method=method)

    delta_cis(params=params, method=method)

    return method
