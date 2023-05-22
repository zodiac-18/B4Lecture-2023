"""Main file."""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import linear_regression as lr


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="Perform regression analysis.")
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
    """Read csv file.

    Args:
        file_path (str): Csv file to read

    Returns:
        ndarray: Data read
    """
    data_set = np.loadtxt(fname=file_path, delimiter=",", skiprows=1)

    return data_set


def scat_plot2d(x, y, beta_r, beta):
    """Plot two-dimensional data.

    Args:
        x (ndarray): data of x axis
        y (ndarray): data of y axis
        beta_r (ndarray): Regularized regression coefficient
        beta (ndarray): regression coefficient
    """
    x_line = np.linspace(min(x), max(x), 100)
    y_expect = lr.prediction2d(beta, x_line)
    y_r = lr.prediction2d(beta_r, x_line)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c="red", label="Original data")
    ax.plot(x_line, y_expect, label=label2d(beta))
    ax.plot(x_line, y_r, color="green", label=label2d(beta_r))
    ax.set_title("Simple Linear Regression")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.savefig("result/simple_lr.png")
    plt.show()


def scat_plot3d(x, y, z, beta, beta_r, N1, N2):
    """Plot three-dimensional data.

    Args:
        x (ndarray): data of x axis
        y (ndarray): data of y axis
        z (ndarray): data of z axis
        beta_r (ndarray): Regularized regression coefficient
        beta (ndarray): regression coefficient
        N1 (int) : dimension of x
        N2 (int) : dimension of y
    """
    x_line = np.linspace(min(x), max(x), 100)
    y_line = np.linspace(min(y), max(y), 100)
    x_mesh, y_mesh = np.meshgrid(x_line, y_line)
    z_expect = lr.prediction3d(beta, x_mesh, y_mesh, N1, N2)
    z_r = lr.prediction3d(beta_r, x_mesh, y_mesh, N1, N2)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_xlabel("x", fontsize=10)
    ax1.set_ylabel("y", fontsize=10)
    ax1.set_zlabel("z", fontsize=10)
    ax1.set_title("Multiple Linear Regression")
    ax1.scatter(x, y, z, marker="o", c="red", label="Original data")
    ax1.plot_wireframe(
        x_mesh,
        y_mesh,
        z_expect,
        color="blue",
        label=label3d(beta, N1),
    )
    ax1.legend()

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_xlabel("x", fontsize=10)
    ax2.set_ylabel("y", fontsize=10)
    ax2.set_zlabel("z", fontsize=10)
    ax2.set_title("Multiple Linear Regression(regularization)")
    ax2.scatter(x, y, z, marker="o", c="red", label="Original data")
    ax2.plot_wireframe(
        x_mesh,
        y_mesh,
        z_r,
        color="green",
        label=label3d(beta_r, N1),
    )
    ax2.legend()
    plt.savefig("result/multiple_lr.png")
    plt.show()


def label2d(beta):
    """Generate label of data1.csv and data2.csv.

    Args:
        beta (np.ndarray): regression coefficient
    Returns:
        string: function equation
    """
    equation = "y="
    for i, b in enumerate(beta):
        b = b[0]
        if i == 0:
            equation += f"{b:+.3f}"
        else:
            equation += f"{b:+.3f}$x^{i}$"
    return equation


def label3d(beta, N1):
    """Generate label.

    Args:
        beta (np.ndarray): regression coefficient
        N1 (int): dimension of x axis
    Returns:
        string: function equation
    """
    equation = "y="
    for i, b in enumerate(beta):
        b = beta[i]
        b = b[0]
        if i <= N1:
            if i == 0:
                equation += f"{b:.3f}"
            else:
                equation += f"{b:+.2f}$x^{i}$"
        else:
            equation += f"{b:+.2f}$y^{i}$"

    return equation


def main():
    """Regression analysis using the least squares method."""
    args = parse_args()

    file_path = args.csv_file
    N1 = args.dimension1
    N2 = args.dimension2
    lamb = args.parameter

    data = open_csv(file_path)

    if data.shape[1] == 2:
        x, y = data[:, 0], data[:, 1]
        y = y.reshape(-1, 1)
        X = lr.simple_regression(x, N1)
        beta = lr.beta_function(X, y)
        beta_r = lr.regularization(X, y, lamb)
        scat_plot2d(x, y, beta_r, beta)

    elif data.shape[1] == 3:
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        z = z.reshape(-1, 1)
        X = lr.multiple_regression(x, y, N1, N2)
        beta = lr.beta_function(X, z)
        beta_r = lr.regularization(X, z, lamb)
        scat_plot3d(x, y, z, beta, beta_r, N1, N2)


if __name__ == "__main__":
    main()
