"""Linear Regression Modules."""

import numpy as np


def calc_coef(X, y):
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


def model1d(x, N):
    # model1d = beta0 * x**0 + bete1 * x**1 + ... + betaN * x**N
    X = np.concatenate([x**i for i in range(0, N+1)], axis=1)
    return X


def expect1d(beta, x):
    y = np.zeros_like(x)
    for i, b in enumerate(beta):
        y += b * x**i
    return y


def model2d(x1, x2, N1, N2):
    # model2d = beta0 * x1**0
    #         + beta 1    *x1**1 +...+ beta N1    *x1**N1
    #         + beta(N1+1)*x2**1 +...+ beta(N1+N2)*x2**N2
    X = np.concatenate(
        [x1**i for i in range(0, N1+1)] +
        [x2**i for i in range(1, N2+1)], axis=1)
    return X


def expect2d(beta, N1, N2, x1_mesh, x2_mesh):
    assert len(beta) == N1 + N2 + 1

    y = np.zeros_like(x1_mesh)
    for i in range(0, N1+1):
        y += beta[i] * x1_mesh**i
    for i in range(1, N2+1):
        y += beta[N1+i] * x2_mesh**i
    return y
