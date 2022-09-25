from src.basis import bernstein as basis


def predict(gamma, k, from_, to_):
    gamma = gamma.reshape(-1, 1)
    basis_ = basis.integrate(k=k, a=from_, b=to_)
    return basis_ @ gamma
