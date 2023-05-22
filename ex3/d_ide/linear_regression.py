"""Linear regression file."""
import numpy as np


def simple_regression(x, N):
    """Calculate of single regression.

    Args:
        x (ndarray): data of x axis
        N (int): dimension of x

    Returns:
        ndarray: matrix of X
    """
    X = np.zeros((len(x), N + 1))

    for i in range(0, len(x)):
        for n in range(0, N + 1):
            X[i][n] = x[i] ** n

    return X


def beta_function(X, y):
    """Calculate regression coefficients.

    Args:
        X (ndarray): data of x
        y (ndarray): data of y

    Returns:
        ndarray: matrix of regression coefficients
    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


def regularization(X, y, lamb):
    """Calculate regularization regression coefficients.

    Args:
        X (ndarray): data of x
        y (ndarray): data of y
        lamb (ndarray): parameter

    Returns:
        ndarray: matrix of regularization regression coefficients
    """
    beta = np.linalg.inv(X.T @ X + lamb * np.identity(X.shape[1])) @ X.T @ y
    return beta


def prediction2d(beta, x):
    """Predicting the equation of two-dimension.

    Args:
        beta (ndarray): regression coefficient
        x (ndarray): data of x axis

    Returns:
        ndarray: predicted equation
    """
    y = np.zeros_like(x)
    for i, b in enumerate(beta):
        y += b * x**i

    return y


def multiple_regression(x, y, N1, N2):
    """Calculate of multiple regression.

    Args:
        x (ndarray): data of x axis
        y (ndarray): data of y axis
        N1 (int): dimension of x
        N2 (int): dimension of y

    Returns:
        ndarray: matrix of X
    """
    X = np.zeros((len(x), N1 + N2 + 1))

    for i in range(0, len(x)):
        for n in range(0, N1 + 1):
            X[i][n] = x[i] ** n
        for k in range(1, N2 + 1):
            X[i][k + N1] = y[i] ** k

    return X


def prediction3d(beta, x, y, N1, N2):
    """Predicting the equation of two-dimension.

    Args:
        beta (ndarray): regression coefficient
        x (ndarray): data of x axis
        y (ndarray): data of y axis
        N1 (int): dimension of x
        N2 (int): dimension of y

    Returns:
        ndarray: predicted equation
    """
    z = np.zeros_like(x)
    for i in range(0, N1 + 1):
        z += beta[i] * x**i
    for i in range(1, N2 + 1):
        z += beta[N1 + i] * y**i

    return z
