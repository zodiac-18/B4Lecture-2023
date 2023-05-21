import numpy as np

def simple_regression(x, N):
    X = np.zeros((len(x) ,N+1))

    for i in range(0, len(x)):
      for n in range(0, N+1):
        X[i][n] = x[i]**n

    return X


def beta_function(X, y):
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


def regularization(X, y, lamb):
    beta = np.linalg.inv(X.T @ X + lamb * np.identity(X.shape[1])) @ X.T @ y
    return beta


def expect(beta, x):
    y = np.zeros_like(x)
    for i, b in enumerate(beta):
        y += b * x**i

    return y


def multiple_regression(x, y, N1, N2):

    X = np.zeros((len(x) ,N1+N2+1))

    for i in range(0, len(x)):
      for n in range(0, N1+1):
        X[i][n] = x[i]**n
      for k in range(1, N2+1):
        X[i][k+N1] = y[i]**k

    return X


def expect2d(beta, x, y, N1, N2):
    z = np.zeros_like(x)
    for i in range(0, N1+1):
        z += beta[i] * x**i
    for i in range(1, N2+1):
        z += beta[N1+i] * y**i

    return z