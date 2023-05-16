"""Main 2."""
import matplotlib.pyplot as plt
import numpy as np


def model(x, N):
    """Model function."""
    X = np.concatenate([x**i for i in range(0, N+1)], axis=1)
    return X


def calc_coef(X, y):
    """Calculate coefficients."""
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


def expect(beta, x):
    """Expect function."""
    y = np.zeros_like(x)
    for i, b in enumerate(beta):
        y += b * x**i
    return y


if __name__ == "__main__":

    # load data
    data = np.loadtxt("../data2.csv", delimiter=",", skiprows=1)
    x, y = data[:, 0], data[:, 1]
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)

    # generate model & calcurate coefficients
    X = model(x, 3)
    beta = calc_coef(X, y)

    # draw graph
    plt.scatter(x, y, marker="o", facecolor="None", edgecolors="red")

    # draw graph
    # TODO(label): 軸ラベルなどをつける
    x_axis = np.linspace(min(x), max(x), 100)
    plt.plot(x_axis, expect(beta, x_axis))
    plt.show()
