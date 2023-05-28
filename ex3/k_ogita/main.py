#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Least squares method."""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    """Linear regression model."""

    def __init__(self, degree=1):
        """
        Initialize instance variables.

        Args:
            degree (int, optional): The degree of the polynominal. Defaults to 1.
        """
        self.degree = degree
        self.w = None

    def Polynominal(self, x1, x2=None):
        """
        Create a polynominal to be fitted.

        Args:
            x1 (ndarray): An array of explanatory variables.
            x2 (ndarray, optional): The second array of explanatory variables.
            Used only when the dimension of the data is 3D. Defaults to None.

        Returns:
            ndarray: Polynominal to be fitted.
        """
        degree = self.degree
        power_lis = np.arange(degree + 1)
        # For two-dimension data
        if x2 is None:
            phi_x = np.array(np.power(x1[:, np.newaxis], power_lis))
        # For three-dimension data
        else:
            phi_x = np.column_stack(
                [
                    np.power(x1[:, np.newaxis], power_lis),
                    np.power(x2[:, np.newaxis], power_lis[1:]),
                ]
            )
        return phi_x

    def fit(self, data, norm=False, lamb=0.0):
        """
        Fitting polynomials using the least squares method.

        Args:
            data (ndarray): Input data.
            norm (bool, optional): Whether to apply normalization or not. Defaults to False.
            lamb (float, optional): Normalization factor. Defaults to 0.0.
        """
        dim = len(data)
        degree = self.degree
        # Generate the polynomial to be fitted
        if dim == 2:
            x, y = data[0], data[1]
            phi = self.Polynominal(x)
            eye_mat = np.eye(degree + 1)
        elif dim == 3:
            x1, x2, y = data[0], data[1], data[2]
            phi = self.Polynominal(x1, x2)
            eye_mat = np.eye(2 * degree + 1)

        # Apply normalization if norm is True
        if norm:
            self.w = np.linalg.inv(phi.T @ phi + lamb * eye_mat) @ phi.T @ y
        else:
            self.w = np.linalg.inv(phi.T @ phi) @ phi.T @ y

    def predict(self, x1, x2=None):
        """
        Make predictions based on the results of linear regression.

        Args:
            x1 (ndarray): An array of explanatory variables.
            x2 (ndarray, optional): The second array of explanatory variables.
            Used only when the dimension of the data is 3D. Defaults to None.

        Returns:
            ndarray: Prediction result of y.
        """
        w = self.w
        degree = self.degree
        # For two-dimension　data
        if x2 is None:
            phi = self.Polynominal(x1)
            predicted_y = phi @ w
        # For three-dimension data
        else:
            X1, X2 = np.meshgrid(x1, x2)
            predicted_y = np.power(X1, 0) * w[0]
            for i in range(degree + 1):
                predicted_y += np.power(X1, i) * w[i] + np.power(X2, i) * w[i + degree]
        return predicted_y


def main():
    """Apply linear regression to the input data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to the input data")
    parser.add_argument(
        "-d", "--degree", help="Regression degree", type=int, required=True
    )
    parser.add_argument(
        "-n",
        "--norm",
        help="Whether to apply normalization or not",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--lamb",
        help="Coefficient of normalization",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    # Get the path of data
    path = args.path
    # Get the filename of data from the path
    file_name = os.path.splitext(os.path.basename(path))[0]

    # Read data from the csv file
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(list(map(float, row)))
    data = np.array(data).T
    # Regression degree
    degree = args.degree
    # Whether to apply normalization or not
    norm = args.norm
    # Normalization factor
    norm_lamb = args.lamb

    # Get the dimention of the data.
    dim = len(data)

    # Two-dimensional regression analysis
    if dim == 2:
        x, y = data[0], data[1]
        # Learning data with linear regression analysis
        model = LinearRegression(degree)
        model.fit(data, norm, norm_lamb)

        # Predict y based on learning results
        X = np.linspace(x.min(), x.max())
        predicted_y = model.predict(X)
        w = model.w

        # Create a label for regression curve(line)
        f_label = ""
        for i in reversed(range(degree + 1)):
            if i == 0:
                f_label += f"{w[i]:+.3f}"
            elif i == 1:
                if degree == 1:
                    f_label += f"{w[i]:.3f}$x$"
                else:
                    f_label += f"{w[i]:+.3f}$x$"
            else:
                if i == degree:
                    f_label += f"{w[i]:.3f}$x^{str(i)}$"
                else:
                    f_label += f"{w[i]:+.3f}$x^{str(i)}$"

        # Plot data and regression curve(line)
        plt.plot(X, predicted_y, c="orange", label=f_label)
        plt.scatter(x, y, c="b", label="data")
        plt.title(file_name)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
        plt.tight_layout()
        plt.savefig(f"{file_name}.png")
        plt.show()

    elif dim == 3:
        x1, x2, y = data[0], data[1], data[2]
        # Learning data with linear regression analysis
        model = LinearRegression(degree)
        model.fit(data, norm, norm_lamb)

        # Predict y based on learning results
        x1_lin = np.linspace(x1.min(), x1.max())
        x2_lin = np.linspace(x2.min(), x2.max())
        predicted_y = model.predict(x1_lin, x2_lin)
        w = model.w

        # Create a label for regression plane
        f_label = ""
        # Create a label for x-terms
        for i in reversed(range(1, degree + 1)):
            if i == 1:
                f_label += f"{w[i]:+.3f}$x$"
            else:
                if i == degree:
                    f_label += f"{w[i]:.3f}$x^{str(i)}$"
                else:
                    f_label += f"{w[i]:+.3f}$x^{str(i)}$"
        # Create a label for y-terms
        for i in reversed(range(degree + 1, len(w))):
            if i == degree + 1:
                f_label += f"{w[i]:+.3f}$y$"
            else:
                f_label += f"{w[i]:+.3f}$y^{str(i-degree)}$"
        f_label += f"{w[0]:+.3f}"

        # Plot data and regression plane
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X1, X2 = np.meshgrid(x1_lin, x2_lin)
        ax.scatter(x1, x2, y, c="b", label="data")
        ax.plot_wireframe(X1, X2, predicted_y, color="orange", label=f_label, alpha=0.5)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        ax.set_title(file_name)
        plt.legend(loc=0)
        plt.savefig(f"{file_name}.png")
        plt.show()


if "__main__" == __name__:
    main()
