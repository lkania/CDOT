import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 600
from src.bin import proportions, uniform_bin
import jax.numpy as np
from src.predict import predict


def histogram(X, from_, to_):
    props = proportions(X=X, from_=from_, to_=to_)
    plt.hist(from_, to_, weights=props, density=False)
    plt.show(block=True)


def uniform_histogram(X, step=0.005):
    from_, to_ = uniform_bin(step=step)
    histogram(X, from_, to_)


def density(X, lower, upper, methods, step=None):
    if step is not None:
        from_, to_ = uniform_bin(step=step)
    else:
        from_ = methods[0].from_
        to_ = methods[0].to_

    props = proportions(X=X, from_=from_, to_=to_)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    ax.hist(from_, to_, weights=props, density=False, color='blue', alpha=0.2)
    ax.axvline(x=lower, color='green', linestyle='--')
    ax.axvline(x=upper, color='green', linestyle='--')

    for method in methods:
        preds = predict(method.gamma, method.k, from_, to_).reshape(-1)
        n, bins, patches = \
            ax.hist(from_, to_, weights=preds, density=False,
                    color=method.color, histtype='step', label=method.name)
        patches[0].set_xy(patches[0].get_xy()[1:-1])

    ax.legend()
    # approximation via density function
    # x = np.arange(0,1,step=step)
    # dens = density_(X=x, k=method.k, gamma=method.gamma)
    # dens = dens * step
    # ax.plot(x,dens,color='red',lw=2)

    plt.show(block=True)


def residuals(X, lower, upper, methods, step=None, size=2):
    if step is not None:
        from_, to_ = uniform_bin(step=step)
    else:
        from_ = methods[0].from_
        to_ = methods[0].to_

    props = proportions(X=X, from_=from_, to_=to_)
    x = (from_ + to_) / 2

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for method in methods:
        preds = predict(method.gamma, method.k, from_, to_).reshape(-1)
        errors = props - preds
        ax.scatter(x, errors, color=method.color, label=method.name, s=size)

    ax.axvline(x=lower, color='green', linestyle='--')
    ax.axvline(x=upper, color='green', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylabel('X_obs-X_hat')
    ax.set_xlabel('Energy spectrum')

    plt.show(block=True)


def error(method):
    errors = method.errors(method.gamma)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.scatter(np.arange(1, len(errors) + 1), errors)

    if "bins" in method:
        ax.axvline(x=int(method.bins / 2), color='red', linestyle='--')

    plt.show(block=True)


def matrix(m, colLabels, rowLabels):
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=m, colLabels=colLabels, rowLabels=rowLabels, loc='center')

    fig.tight_layout()
    plt.show(block=True)
    return fig


def title(ax, str):
    ax.set_title(str)


def remove_y_axis(ax):
    ax.get_yaxis().set_visible(False)


def save(fig, path):
    fig.savefig(fname=path, bbox_inches='tight')
    plt.close(fig)
