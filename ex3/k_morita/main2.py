"""Main 2."""
import argparse
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np

import linear_regression as mylr


if __name__ == "__main__":
    # args parser settings
    parser = argparse.ArgumentParser(description="Process linear-regression")
    parser.add_argument("-a", "--alpha", type=np.float64, default=1)

    # get args
    args = parser.parse_args()
    alpha = args.alpha

    # load data
    data = np.loadtxt("../data2.csv", delimiter=",", skiprows=1)
    x, y = data[:, 0], data[:, 1]
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)

    # generate model & calcurate coefficients
    N = 3
    X = mylr.model1d(x, N)
    beta = mylr.calc_coef(X, y)
    beta_with_regular = mylr.calc_coef_with_regularization(X, y, alpha)

    # draw graph
    x_axis = np.linspace(min(x), max(x), 100)
    plt.scatter(x, y, facecolor="None", edgecolors="red", label="Observed")
    plt.plot(x_axis, mylr.expect1d(beta, x_axis), label=mylr.label1d(beta))
    plt.plot(
        x_axis,
        mylr.expect1d(beta_with_regular, x_axis),
        label=mylr.label1d(beta_with_regular) + "(正則化)",
    )
    plt.title("Polynomial Linear-Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
