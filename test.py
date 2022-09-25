from jaxopt import FixedPointIteration
import jax.numpy as np
import jax


def T(x, theta):  # contractive map
    return 0.5 * x + theta


fpi = FixedPointIteration(fixed_point_fun=T)
x_init = np.array(0.)
theta = np.array(0.5)


def fixed_point(x, theta):
    return fpi.run(x, theta).params


print(jax.grad(fixed_point, argnums=1)(x_init, theta))  # only gradient
print(jax.value_and_grad(fixed_point, argnums=1)(x_init, theta))  # both value and gradient
