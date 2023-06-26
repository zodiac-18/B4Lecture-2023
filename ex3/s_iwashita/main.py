"""Regression analysis using the least squares method."""
import argparse

import matplotlib.pyplot as plt
import numpy as np


def least_squares_method(x, y, order: int, dimension: int, lamb: int):
    """Least-squares method.

    Args:
        x (ndarray): Explanatory variables
        y (ndarray): Objective variables
        order (int): Order of regression equation
        dimension (int): Dimensions of the given data
        lamb (int): Regularization factor

    Returns:
        u (ndarray): Regression coefficient
        u_re (ndarray): Regularized regression coefficient
    """
    A_t = x**order
    for i in range(order - 1, -1, -1):
        A_t = np.vstack((A_t, x**i))
    if dimension == 3:
        A_t = np.delete(A_t, slice(-1, -dimension + 1, -1), 0)
    A = A_t.T
    v = np.array(y).T
    u = np.linalg.inv(A.T @ A) @ A.T @ v
    u_reg = np.linalg.inv(A.T @ A + lamb * np.identity(A.shape[1])) @ A.T @ v

    return u, u_reg


def generate_plot_data(x, u, order, dimension):
    """Generate plot data.

    Args:
        x (ndarray): Explanatory variables
        u (ndarray): Regression coefficient
        order (int): Order of regression equation
        dimension (int): Dimensions of the given data

    Returns:
        ndarray: Plot data
    """
    if dimension == 2:
        x_re = np.linspace(min(x), max(x), 100)
        y_re = np.zeros_like(x_re)
        for i in range(order + 1):
            y_re += u[i] * x_re**i
        z_re = None

    elif dimension == 3:
        x_re = np.linspace(min(x[0]), max(x[0]), 10)
        y_re = np.linspace(min(x[1]), max(x[1]), 10)
        x_re, y_re = np.meshgrid(x_re, y_re)
        z_re = np.zeros_like(x_re)
        z_re = u[0] * x_re**0
        for i in range(1, order + 1):
            z_re += u[2 * i - 1] * y_re**i + u[2 * i] * x_re**i

    return x_re, y_re, z_re


def generate_equation(u, order, dimension):
    """Generate label.

    Args:
        u (ndarray): Regression coefficient
        order (int): Order of regression equation
        dimension (int): Dimensions of the given data

    Returns:
        str: label
    """
    if dimension == 2:
        label = f"y={u[0]:.2f}"
        for i in range(1, order + 1):
            if u[i] > 0:
                label += "+"
            if i == 1:
                label += f"{u[i]:.2f}x"
            else:
                label += f"{u[i]:.2f}" + "$x^{" + str(i) + "}$"

    elif dimension == 3:
        label = f"z={u[0]:.2f}"
        for i in range(1, order * 2 + 1):
            if u[i] > 0:
                label += "+"
            if i == 1:
                label += f"{u[i]:.2f}y"
            elif i == 2:
                label += f"{u[i]:.2f}x"
            elif i % 2 == 1:
                label += f"{u[i]:.2f}" + "$y^{" + str(i // 2 + 1) + "}$"
            else:
                label += f"{u[i]:.2f}" + "$x^{" + str(i // 2) + "}$"

    return label


def main():
    """Plot scatter plots and regression equations."""
    parser = argparse.ArgumentParser(
        "Regression analysis using the least squares method"
    )

    parser.add_argument("filename")
    parser.add_argument("order", type=int, help="Order of regression equation")
    parser.add_argument(
        "-l", "--lamb", type=int, default=10, help="Regularization factor"
    )
    parser.add_argument(
        "-f", "--figname", type=str, default="sample.png", help="Figure name"
    )

    args = parser.parse_args()

    filename = args.filename
    order = args.order
    lamb = args.lamb
    figname = args.figname

    # read csv file
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)
    dimension = len(data[0])

    # Separate data into explanatory and objective variables
    x = data[:, 0]
    y = data[:, 1]
    if dimension == 3:
        x1 = x
        x2 = y
        x = np.vstack((x, y))
        y = data[:, 2]

    # Calculate regression coefficient
    u, u_reg = least_squares_method(x, y, order, dimension, lamb)
    u, u_reg = u[::-1], u_reg[::-1]

    # Generate data to plot
    x_re, y_re, z_re = generate_plot_data(x, u, order, dimension)
    x_rre, y_rre, z_rre = generate_plot_data(x, u_reg, order, dimension)
    label = generate_equation(u, order, dimension)
    label_reg = generate_equation(u_reg, order, dimension)

    # Plot data
    plt.figure()
    if dimension == 2:
        ax = plt.subplot()
        ax.scatter(x, y, s=5, c="b", label="Observed data")
        ax.plot(x_re, y_re, c="c", label=label)
        ax.plot(x_rre, y_rre, c="g", label=label_reg)
        ax.set_xlabel("X", size="small")
        ax.set_ylabel("Y", size="small")
        ax.legend()
    elif dimension == 3:
        x1 = x[0]
        x2 = x[1]
        ax1 = plt.subplot(121, projection="3d")
        ax1.scatter(x1, x2, y, c="b", label="Observed data")
        ax1.plot_wireframe(x_re, y_re, z_re, color="c", label=label)
        ax1.set_xlabel("X", size="x-small")
        ax1.set_ylabel("Y", size="x-small")
        ax1.set_zlabel("Z", size="x-small")
        ax1.legend(bbox_to_anchor=(1, 1.1), fontsize="xx-small")
        ax2 = plt.subplot(122, projection="3d")
        ax2.scatter(x1, x2, y, c="b", label="Observed data")
        ax2.plot_wireframe(x_rre, y_rre, z_rre, color="g", label=label_reg)
        ax2.set_xlabel("X", size="x-small")
        ax2.set_ylabel("Y", size="x-small")
        ax2.set_zlabel("Z", size="x-small")
        ax2.legend(bbox_to_anchor=(1, 1.1), fontsize="xx-small")
    plt.savefig(figname)
    plt.show()


if __name__ == "__main__":
    main()
