"""Main 3."""
import matplotlib.pyplot as plt
import numpy as np


def model(x1, x2, N1, N2):
    """Model."""
    # model = beta0
    #         + beta 1    *x1**1 +...+ beta N1    *x1**N1
    #         + beta(N1+1)*x2**1 +...+ beta(N1+N2)*x2**N2
    X = np.concatenate(
        [x1**i for i in range(0, N1+1)] +
        [x2**i for i in range(1, N2+1)], axis=1)
    return X


def calc_coef(X, y):
    """Calculate coefficient."""
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


def expect2d(beta, N1, N2, x1_mesh, x2_mesh):
    """Expect2d."""
    assert len(beta) == N1 + N2 + 1

    y = np.zeros_like(x1_mesh)
    for i in range(0, N1+1):
        y += beta[i] * x1_mesh**i
    for i in range(1, N2+1):
        y += beta[N1+i] * x2_mesh**i

    return y


if __name__ == "__main__":
    # load data
    data = np.loadtxt("../data3.csv", delimiter=",", skiprows=1)
    x1, x2, y = data[:, 0], data[:, 1], data[:, 2]
    x1, x2, y = x1.reshape(-1, 1), x2.reshape(-1, 1), y.reshape(-1, 1)

    N1, N2 = 1, 2
    X = model(x1, x2, N1, N2)
    beta = calc_coef(X, y)

    # draw graph
    x1_axis = np.linspace(min(x1), max(x2), 100)
    x2_axis = np.linspace(min(x2), max(x2), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_axis, x2_axis)
    y_expect = expect2d(beta, N1, N2, x1_mesh, x2_mesh)

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
