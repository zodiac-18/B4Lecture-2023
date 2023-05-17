"""Main 1 ."""
import matplotlib.pyplot as plt
import numpy as np

import linear_regression as mylr


if __name__ == "__main__":

    # load data
    data = np.loadtxt("../data1.csv", delimiter=",", skiprows=1)
    x, y = data[:, 0], data[:, 1]
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)

    # generate model & calcurate coefficients
    N = 1
    X = mylr.model1d(x, N)
    beta = mylr.calc_coef(X, y)

    # draw graph
    x_axis = np.linspace(min(x), max(x), 100)
    plt.scatter(x, y, facecolor="None", edgecolors="red", label="Observed")
    plt.plot(x_axis, mylr.expect1d(beta, x_axis), label=mylr.label1d(beta))
    plt.title("Simple Linear-Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
