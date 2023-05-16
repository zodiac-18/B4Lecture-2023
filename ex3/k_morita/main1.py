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
    X = mylr.model1d(x, 1)
    beta = mylr.calc_coef(X, y)

    # draw graph
    plt.scatter(x, y, marker="o", facecolor="None", edgecolors="red")

    # draw graph
    # TODO(label): 軸ラベルなどをつける
    x_axis = np.linspace(min(x), max(x), 100)
    plt.plot(x_axis, mylr.expect1d(beta, x_axis))
    plt.show()
