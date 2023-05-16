"""Main 3."""
import matplotlib.pyplot as plt
import numpy as np

import linear_regression as mylr


if __name__ == "__main__":
    # load data
    data = np.loadtxt("../data3.csv", delimiter=",", skiprows=1)
    x1, x2, y = data[:, 0], data[:, 1], data[:, 2]
    x1, x2, y = x1.reshape(-1, 1), x2.reshape(-1, 1), y.reshape(-1, 1)

    N1, N2 = 1, 2
    X = mylr.model2d(x1, x2, N1, N2)
    beta = mylr.calc_coef(X, y)

    # draw graph
    x1_axis = np.linspace(min(x1), max(x2), 100)
    x2_axis = np.linspace(min(x2), max(x2), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_axis, x2_axis)
    y_expect = mylr.expect2d(beta, N1, N2, x1_mesh, x2_mesh)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x1", fontsize=22)
    ax.set_ylabel("x2", fontsize=22)
    ax.set_zlabel("y", fontsize=22)
    ax.scatter(x1, x2, y, marker="o", c="red", label="Observed data")
    ax.plot_wireframe(x1_mesh, x2_mesh, y_expect, color="blue")
    ax.legend(fontsize=15)
    plt.tick_params(labelsize=18)
    plt.show()
