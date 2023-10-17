from jax import grad, hessian, numpy as np, vmap, jit
from functools import partial
from jax.lax import stop_gradient


@partial(jit, static_argnames=['signal'])
def objective(X, nu, signal, background_hat):
    X = X.reshape(-1)
    nu = nu.reshape(-1)
    mu_hat = nu[0]
    sigma2_hat = nu[1]
    lambda_hat = nu[2]

    scaled_signal_density = lambda_hat * signal(X=X, mu=mu_hat,
                                                sigma2=sigma2_hat).reshape(-1)
    scaled_background_density = (1 - lambda_hat) * background_hat.reshape(-1)
    ll_hat = np.mean(np.log(scaled_signal_density + scaled_background_density))

    return ll_hat


@partial(jit, static_argnames=['signal'])
def psi(X, nu, signal, background_hat):
    grad_op = grad(
        fun=lambda nu, X, background_hat: objective(X=X, nu=nu, signal=signal,
                                                    background_hat=background_hat),
        argnums=0,
        has_aux=False)
    return grad_op(nu, X, background_hat)


@partial(jit, static_argnames=['signal'])
def psi_per_observation(X, nu, signal, background_hat):
    X = X.reshape(-1)
    background_hat = background_hat.reshape(-1)
    psi_ = vmap(lambda X, background_hat: psi(X=X, nu=nu, signal=signal,
                                              background_hat=background_hat))
    return psi_(X, background_hat)


@partial(jit, static_argnames=['signal'])
def Psi(X, nu, signal, background_hat):
    nu = nu.reshape(-1)
    psis = psi_per_observation(X=X, nu=nu, signal=signal,
                               background_hat=background_hat)
    psis = np.transpose(psis)  # shape: n_parameters x n_obs
    mean_psis = np.mean(psis, axis=1).reshape(-1)

    return mean_psis, psis


def estimate(h_hat, method, params):
    lambda_hat0, aux = method.background.estimate_lambda(h_hat)
    gamma_hat, gamma_aux = aux

    # fake output to avoid doing signal estiamtion
    # uncomment to estimate signal
    nu_hat = np.array([0.0, 0.0, 0.0])
    signal_aux = (0.0, 0.0)
    mean_psis = np.array([0.0, 0.0, 0.0])
    psis = 0.0
    background_hat = 0.0

    # background_hat = method.background.estimate_background_from_gamma(
    #     gamma=gamma_hat, X=method.X)
    # nu_hat, signal_aux = method.signal.estimate_nu(lambda_hat0=lambda_hat0,
    #                                                background_hat=background_hat)
    # mean_psis, psis = Psi(X=method.X,
    #                       nu=stop_gradient(nu_hat),
    #                       # stop_gradient so that the derivative w.r.t. h_hat
    #                       # goes only through background_hat
    #                       signal=params.signal.signal,
    #                       background_hat=background_hat)
    aux = (
        lambda_hat0, nu_hat, gamma_hat, background_hat, psis, gamma_aux,
        signal_aux)
    return np.insert(mean_psis, obj=0, values=lambda_hat0), aux


def influence(method, params):
    influence_, aux = method.background.influence(
        func=partial(estimate, method=method, params=params))

    lambda_hat0, nu_hat, gamma_hat, background_hat, psis, gamma_aux, signal_aux = aux

    influence_lambda_hat0 = influence_[0, :].reshape(1, -1)  # 1 x n_obs
    influence_nu_hat = influence_[1:, :]  # shape: n_parameters x n_obs

    # Disabled to speed_up computation
    # influence_nu_hat = influence_nu_hat + psis  # shape: n_parameters x n_obs
    # hessian_op = hessian(
    #     fun=lambda nu: objective(X=method.X, nu=nu,
    #                              signal=params.signal.signal,
    #                              background_hat=background_hat),
    #     argnums=0,
    #     has_aux=False)
    # info_matrix = hessian_op(nu_hat)  # shape = (n_parameters, n_parameters)
    # inv_info_matrix = np.linalg.pinv(info_matrix)
    # influence_nu_hat = inv_info_matrix @ influence_nu_hat

    influence_ = np.vstack((influence_lambda_hat0,
                            influence_nu_hat))  # shape = (n_parameters+1,n_obs)
    aux = (np.insert(nu_hat.reshape(-1), obj=0, values=lambda_hat0), gamma_hat,
           gamma_aux, signal_aux)
    return influence_, aux
