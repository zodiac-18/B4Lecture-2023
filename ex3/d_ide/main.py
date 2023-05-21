import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import linear_regression as lr


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="data csv file",
    )
    parser.add_argument(
        "--dimension1",
        type=int,
        default=1,
        help="dimension",
    )
    parser.add_argument(
        "--dimension2",
        type=int,
        default=1,
        help="dimension",
    )
    parser.add_argument(
        "--parameter",
        type=int,
        default=10,
        help="Loss function parameter",
    )

    return parser.parse_args()


def open_csv(file_path):
    data_set = np.loadtxt(
        fname=file_path,
        delimiter=",",
        skiprows=1
    )

    return data_set

def scat_plot2d(x, y, beta_r, beta):
    x_line = np.linspace(min(x), max(x), 100)
    y_expect = lr.expect(beta, x_line)
    y_r = lr.expect(beta_r, x_line)
    fig, ax = plt.subplots()
    ax.scatter(x, y, facecolor="None", edgecolors="red", label="Observed")
    ax.plot(x_line, y_expect, label=label1d(beta))
    ax.plot(x_line, y_r, label=label1d(beta_r))
    ax.set_title("Simple Linear-Regression")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc=0)
    fig.tight_layout()
    plt.show()


def scat_plot3d(x, y, z, beta, beta_r, N1, N2):
    x_line = np.linspace(min(x), max(x), 100)
    y_line = np.linspace(min(y), max(y), 100)
    x_mesh, y_mesh = np.meshgrid(x_line, y_line)
    z_expect = lr.expect2d(beta, x_mesh, y_mesh, N1, N2)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x", fontsize=22)
    ax.set_ylabel("y", fontsize=22)
    ax.set_zlabel("z", fontsize=22)
    ax.scatter(x, y, z, marker="o", c="red", label="Observed data")
    ax.plot_wireframe(
        x_mesh,
        y_mesh,
        z_expect,
        color="blue",
        label=label2d(beta, 1, 2),
    )
    ax.legend()
    plt.show()

    z_r = lr.expect2d(beta_r, x_mesh, y_mesh, N1, N2)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x", fontsize=22)
    ax.set_ylabel("y", fontsize=22)
    ax.set_zlabel("z", fontsize=22)
    ax.scatter(x, y, z, marker="o", c="red", label="Observed data")
    ax.plot_wireframe(
        x_mesh,
        y_mesh,
        z_r,
        color="blue",
        label=label2d(beta_r, 1, 2),
    )
    ax.legend()
    plt.show()


def label1d(beta):
    """Generate label.
    Args:
        beta (np.ndarray): coefficient vector
    Returns:
        string: function label
    """
    equation = "y="
    for i, b in enumerate(beta):
        b = b[0]
        if i == 0:
            equation += f'{b:+.3f}'
        else:
            equation += f'{b:+.3f}$x^{i}$'
    return equation


def label2d(beta, N1, N2):
    """Generate label.
    Args:
        beta (np.ndarray): coefficient vector
        N1 (int): dimension of x1
        N2 (int): dimension of x2
    Returns:
        string: function label
    """
    equation = "y="
    for i, b in enumerate(beta):
      b = beta[i]
      b = b[0]
      if i <= N1:
        if i == 0:
          equation += f'{b:.3f}'
        else:
          equation += f'{b:+.2f}$x^{i}$'
      else:
        equation += f'{b:+.2f}$y^{i}$'

    return equation


def main():
    args = parse_args()

    file_path = args.csv_file
    N1 = args.dimension1
    N2 = args.dimension2
    lamb = args.parameter

    data = open_csv(file_path)

    if data.shape[1] == 2:
        x, y = data[: ,0], data[ : , 1]
        y = y.reshape(-1, 1)
        X = lr.simple_regression(x, N1)
        beta = lr.beta_function(X, y)
        beta_r = lr.regularization(X, y, lamb)
        scat_plot2d(x, y, beta_r, beta)

    elif data.shape[1] == 3:
        x, y, z = data[: ,0], data[ : , 1], data[: , 2]
        z = z.reshape(-1, 1)
        X =lr.multiple_regression(x, y, N1, N2)
        beta = lr.beta_function(X, z)
        beta_r = lr.regularization(X, z, lamb)
        scat_plot3d(x, y, z, beta, beta_r, N1, N2)

if __name__ == "__main__":
    main()