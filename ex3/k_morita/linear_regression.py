"""Linear Regression Modules."""

import numpy as np


def calc_coef(X, y):
    """Calculate coefficients.

    Args:
        X (np.ndarray): model matrix
        y (np.ndarray): object variable vector

    Returns:
        np.ndarray: coefficient vector
    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


def model1d(x, N):
    """Generate Model.

    Args:
        x (np.ndarray): explanatory variable vector
        N (int): dimension

    Returns:
        np.ndarray: model matrix
    """
    # model1d = beta0 * x**0 + bete1 * x**1 + ... + betaN * x**N
    X = np.concatenate([x**i for i in range(0, N+1)], axis=1)

    return X


def expect1d(beta, x):
    """Calculate object variable with model and coefficients.

    Args:
        beta (np.ndarray): coefficient vector
        x (np.ndarray): explanatory variable vector

    Returns:
        np.ndarray: expected expanatory variable vector
    """
    y = np.zeros_like(x)
    for i, b in enumerate(beta):
        y += b * x**i

    return y


def model2d(x1, x2, N1, N2):
    """Generate model.

    Args:
        x1 (np.ndarray): explanatory variable vector
        x2 (np.ndarray): explanatory variable vector
        N1 (int): dimension of x1
        N2 (int): dimention of x2

    Returns:
        np.ndarray: model matrix
    """
    # model2d = beta0 * x1**0
    #         + beta 1    *x1**1 +...+ beta N1    *x1**N1
    #         + beta(N1+1)*x2**1 +...+ beta(N1+N2)*x2**N2
    X = np.concatenate(
        [x1**i for i in range(0, N1+1)] +
        [x2**i for i in range(1, N2+1)], axis=1)

    return X


def expect2d(beta, N1, N2, x1_mesh, x2_mesh):
    """Calculate object variable with model and coefficients.

    Args:
        beta (np.ndarray): coefficient vector
        N1 (int): dimension of x1
        N2 (int): dimension of x2
        x1_mesh (np.ndarray): explanatory variable matrix
        x2_mesh (np.ndarray): explanatory variable matrix

    Returns:
        np.ndarray: expected object variable matrix
    """
    assert len(beta) == N1 + N2 + 1

    y = np.zeros_like(x1_mesh)
    for i in range(0, N1+1):
        y += beta[i] * x1_mesh**i
    for i in range(1, N2+1):
        y += beta[N1+i] * x2_mesh**i
    return y
