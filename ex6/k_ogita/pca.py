#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Principal Component analysis."""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation as animation


class PCA:
    """Principal Component analysis."""

    def __init__(self, num_prim_comp=None):
        """
        Initialize instance variables.

        Args:
            num_prim_comp (int, optional): Number of eigenvectors to be extracted in fit.
        """
        self.num_prim_comp = num_prim_comp
        self.eigin_value = None
        self.eigin_vector = None
        self.w = None

    def fit(self, data):
        """
        Learning weights using principal component analysis.

        Args:
            data (ndarray): Input data.

        Returns:
            ndarray: Matrix of computed weights.
        """
        # Step1. Standardize data
        data_std = self.standardize(data)
        # Step2. Create a convariance matrix
        cov_mat = np.cov(data_std.T)
        # Step3.　Obtain eigenvalues and eigenvectors
        self.eigin_value, self.eigin_vector = np.linalg.eig(cov_mat)
        # Step4 & 5. Select k eigenvectors in order of increasing eigenvalue
        self.eigin_value = np.sort(self.eigin_value)[::-1]
        self.eigin_vector = self.eigin_vector[:, np.argsort(-self.eigin_value)]
        # Step6. Create projection matrix from eigenvectors
        self.w = self.eigin_vector[:, : self.num_prim_comp]
        return self.w

    def transform(self, data):
        """
        Compress data using calculated weights.

        Args:
            data (ndarray): Input data.

        Returns:
            ndarray: Transformed data.
        """
        # Step7. Obtain feature subspaces using projection matrices
        return data @ self.w

    def contribution_rate(self):
        """
        Calculate contribution rate.

        Returns:
            tuple: Contribution rate and cumulative contribution rate.
        """
        contribution_rate = self.eigin_value / np.sum(self.eigin_value)
        cumulative_contribution_rate = np.cumsum(contribution_rate)
        return contribution_rate, cumulative_contribution_rate

    def standardize(self, data):
        """
        Standardize data.

        Args:
            data (ndarray): Input data.

        Returns:
            ndarray: Standardized data.
        """
        data_std = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        return data_std


def main():
    """Compress data using principal component analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to the input data")
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
    data = np.array(data)
    dim = data.shape[1]
    model = PCA()
    w = model.fit(data)
    comp_data = model.transform(data)
    con_rate, cum_con_rate = model.contribution_rate()
    print("Principal Component Analysis Results")
    print("--------Contribution rate--------")
    [print(f"e{i+1}: {con_rate[i]:.7f}") for i in range(dim)]
    if dim == 2:
        fig1 = plt.figure(figsize=(15, 10))
        ax1 = fig1.add_subplot(111)
        ax1.scatter(data[:, 0], data[:, 1], label="original data")
        ax1.axline(
            (0, 0), w[0], color="red", label=f"Contribution rate:{con_rate[0]:.3f}"
        )
        ax1.axline(
            (0, 0), w[1], color="orange", label=f"Contribution rate:{con_rate[1]:.3f}"
        )
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_title("")
        ax1.legend(loc="upper left")
        ax1.set_title(f"Basis of principal components of {file_name}")
        plt.tight_layout()
        fig1.savefig(f"{file_name}_pca.png")

    elif dim == 3:
        fig1 = plt.figure(figsize=(15, 10))
        ax1 = fig1.add_subplot(111, projection="3d")
        slope_xy = w[1] / w[0]
        slope_yz = w[2] / w[0]
        x = np.linspace(np.min(data[:, 0]), np.max(data[:, 1]), 100)

        def animation_init():
            """
            Initialize plot for drawing animation.

            Returns:
                matplotlib.figure.Figure : Figure.
            """
            ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c="blue", label="data")
            ax1.plot(
                x,
                slope_xy[0] * x,
                slope_yz[0] * x,
                color="red",
                label=f"Contribution rate:{con_rate[0]:.3f}",
            )
            ax1.plot(
                x,
                slope_xy[1] * x,
                slope_yz[1] * x,
                color="orange",
                label=f"Contribution rate:{con_rate[1]:.3f}",
            )
            ax1.plot(
                x,
                slope_xy[2] * x,
                slope_yz[2] * x,
                color="yellow",
                label=f"Contribution rate:{con_rate[2]:.3f}",
            )
            ax1.view_init(elev=0, azim=60, roll=5)
            return (fig1,)

        rotate_elev = np.linspace(0, 60, 120, endpoint=False)
        rotate_azim = np.linspace(60, 420, 120)

        def animate(i):
            """
            Set the viewpoint of the animation.

            Returns:
                matplotlib.figure.Figure : Figure.
            """
            ax1.view_init(elev=rotate_elev[i], azim=rotate_azim[i], roll=5)
            return (fig1,)

        ani = animation(
            fig1,
            func=animate,
            init_func=animation_init,
            frames=np.arange(60),
            interval=100,
            blit=True,
            repeat=False,
        )
        ax1.set_title(f"Basis of principal components of {file_name}")
        ax1.legend()
        ani.save(f"{file_name}_pca.gif", writer="pillow")

        fig2 = plt.figure(figsize=(15, 10))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(comp_data[:, 0], comp_data[:, 1], c="blue", label="compressed data")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title(f"Compressed data of {file_name}.csv")
        ax2.legend(loc="upper left")
        plt.tight_layout()
        fig2.savefig(f"{file_name}_comp.png")
    else:
        order = np.min(np.where(cum_con_rate >= 0.9))
        print("Order:", order)
        x = np.linspace(0, dim, dim)
        fig1 = plt.figure(figsize=(15, 10))
        ax1 = fig1.add_subplot(111)
        ax1.plot(x, cum_con_rate, color="blue", label="total contribution rate")
        ax1.axhline(y=0.9, color="red", label="threshold 90%")
        ax1.axvline(
            x=order,
            color="orange",
            label=f"order of when the cumulative contribution reaches 90% (order={order})",
        )
        ax1.set_xlabel("Number of principal components")
        ax1.set_ylabel("Contribution rate")
        ax1.set_xlim(0, dim)
        ax1.set_title(f"Total contribution rate of {file_name}")
        ax1.legend(loc="upper left")
        fig1.savefig(f"{file_name}_con_rate.png")


if __name__ == "__main__":
    main()
