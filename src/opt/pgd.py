import jax.numpy as np


def nnls(A, b, maxiter=1000):
    b_ = b.reshape(-1, 1)
    AA = A.transpose() @ A
    norm = np.linalg.norm(AA, 2)
    THETA = np.eye(AA.shape[0]) - AA / norm
    theta = A.transpose() @ b_ / norm

    kr = 0
    x = np.zeros((A.shape[1], 1))
    y = x
    error_ = -1
    for k in (np.arange(maxiter) + 1):
        x_ = x
        beta = (k - kr - 1) / (k - kr + 2)
        x = np.maximum(THETA @ y + theta, 0)
        y = x + beta * (x - x_)

        error = np.sum(np.square(A @ x - b_))
        if k > 1 and error > error_:
            x = np.maximum(THETA @ x_ + theta, 0)
            error = np.sum(np.square(A @ x - b_))

            y = x
            kr = kr + k

        error_ = error

    return x, error_
