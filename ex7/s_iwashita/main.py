"""Perform fitting of data using GMM."""
import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GMM:
    """Gausssian Mixture Model."""

    def __init__(self, data, k, filename):
        """Initialize instance variable.

        Args:
            data (ndarray): Input data.
            k (int): Number of clusters.
            filename (str): File name for reading data.
        """
        if len(data.shape) == 1:
            self.data = data[:, None]
        else:
            self.data = data
        self.K = k
        self.filename = filename

    def fit(self):
        """Perform fitting."""
        self.initialize()
        self.EM_algorithm()
        self.display()

    def initialize(self):
        """Initialize variable for EM-algorithm."""
        self.N = self.data.shape[0]
        self.D = self.data.shape[1]
        self.mu = np.random.randn(self.K, self.D)
        self.sigma = np.array([np.identity(self.D) for i in range(self.K)])
        self.pi = np.ones(self.K) / self.K

    def calc_gaussian(self, x=None):
        """Caluculate gaussiam.

        Args:
            x (ndarray, optional): Input data. Defaults to None.

        Returns:
            ndarray: Gaussian.
        """
        if x is None:
            data = self.data
        elif len(x.shape) == 1:
            data = x[:, None]
        else:
            data = x
        diff_data = data[None, :, :] - self.mu[:, None, :]
        tmp = diff_data @ np.linalg.pinv(self.sigma) @ diff_data.transpose(0, 2, 1)
        numerator = np.exp(-0.5 * np.diagonal(tmp, axis1=1, axis2=2))
        dinominator = np.sqrt(
            (2 * np.pi) ** (self.D) * np.linalg.det(self.sigma)
        ).reshape(-1, 1)
        return numerator / dinominator

    def calc_mixture_gaussian(self, x=None):
        """Calculate mixture gaussian.

        Args:
            x (ndarray, optional): Input data. Defaults to None.

        Returns:
            ndarray: Mixture gaussian.
        """
        self.w_gaussian = self.calc_gaussian(x) * self.pi.reshape(-1, 1)
        self.m_gaussian = np.sum(self.w_gaussian, axis=0)
        return self.m_gaussian

    def calc_loglikelihood(self):
        """Calculate log-likelihood."""
        self.calc_mixture_gaussian()
        self.loglikelihood = np.sum(np.log(self.m_gaussian))
        self.loglikelihoods.append(self.loglikelihood)

    def EM_algorithm(self, epsilon=1.0e-4):
        """Perform EM-algorithm.

        Args:
            epsilon (float, optional): Threshold of EM-algorithm. Defaults to 1.0e-4.
        """
        diff = 1
        self.loglikelihoods = []
        self.loglikelihood = 1e5
        while diff > epsilon:
            # ==================== E-step ====================
            # Recalculate the log-likelihood
            pre_loglikelihood = self.loglikelihood
            self.calc_loglikelihood()
            diff = np.abs(self.loglikelihood - pre_loglikelihood)

            # Caluculate the burden ratio gamma
            gamma = self.w_gaussian / self.m_gaussian

            # ==================== M-step ====================
            # Recalculate each variable
            N_k = np.sum(gamma, axis=1)
            self.pi = N_k / self.N
            self.mu = np.sum(gamma[:, :, None] * self.data / N_k[:, None, None], axis=1)
            diff_data = self.data[None, :, :] - self.mu[:, None, :]
            self.sigma = (
                gamma[:, None, :] * diff_data.transpose(0, 2, 1) @ diff_data
            ) / N_k[:, None, None]

    def display(self):
        """Display graphs."""
        if self.D == 1:
            fig1 = plt.figure(figsize=(12, 5))
            ax1 = fig1.add_subplot(121)
            # Display log-likelihood
            ax1.plot(self.loglikelihoods)
            ax1.set(
                xlabel="Iteration",
                ylabel="Log-likelihood",
                title="{} Log-likelihood".format(self.filename[:5]),
            )
            # Display gmm
            ax2 = fig1.add_subplot(122)
            # Display data
            ax2.scatter(
                self.data[:, 0],
                np.zeros(self.data[:, 0].shape[0]),
                c="b",
                marker="$o$",
                label="Data",
            )
            # Display centroid
            ax2.scatter(
                self.mu[:, 0],
                np.zeros(self.mu.shape[0]),
                c="r",
                marker="$x$",
                label="Centroid",
            )
            # Display mixed Gaussian distributions
            x = np.linspace(np.min(self.data[:, 0]), np.max(self.data[:, 0]), 100)
            self.calc_mixture_gaussian(x)
            ax2.plot(x, self.m_gaussian, c="y", label="GMM")
            ax2.set(
                xlabel="x",
                ylabel="Probability density",
                title="Probability density of {}(K = {})".format(
                    self.filename[:5], self.K
                ),
            )
            ax2.grid()
            ax2.legend()
            fig1.savefig("result/{}_result_K{}".format(self.filename[:5], self.K))

        if self.D == 2:
            fig1 = plt.figure(figsize=(12, 5))
            ax1 = fig1.add_subplot(121)
            # Display log-likelihood
            ax1.plot(self.loglikelihoods)
            ax1.set(
                xlabel="Iteration",
                ylabel="Log-likelihood",
                title="{} Log-likelihood".format(self.filename[:5]),
            )
            # Display gmm
            ax2 = fig1.add_subplot(122)
            # Display data
            ax2.scatter(
                self.data[:, 0], self.data[:, 1], c="b", marker="$o$", label="Data"
            )
            # Display centroid
            ax2.scatter(
                self.mu[:, 0], self.mu[:, 1], c="r", marker="$x$", label="Centroid"
            )
            # Display mixed Gaussian distributions
            x1 = np.linspace(np.min(self.data[:, 0]), np.max(self.data[:, 0]), 100)
            x2 = np.linspace(np.min(self.data[:, 1]), np.max(self.data[:, 1]), 100)
            X1, X2 = np.meshgrid(x1, x2)
            X = np.dstack((X1, X2))
            pos = np.array([self.calc_mixture_gaussian(x) for x in X])
            ax2.contour(X1, X2, pos, cmap="jet")
            ax2.set(
                xlabel="x",
                ylabel="y",
                title="Coutour map of {}(K = {})".format(self.filename[:5], self.K),
            )
            ax2.grid()
            ax2.legend()
            fig1.savefig("result/{}_result_K{}".format(self.filename[:5], self.K))

            # Display GMM animation
            fig2 = plt.figure()
            ax = fig2.add_subplot(111, projection="3d")
            # Display data
            ax.scatter(
                self.data[:, 0],
                self.data[:, 1],
                np.zeros(self.data.shape[0]),
                c="b",
                marker="$o$",
                label="Data",
            )
            # Display centroid
            ax.scatter(
                self.mu[:, 0],
                self.mu[:, 1],
                np.zeros(self.mu.shape[0]),
                c="r",
                marker="$x$",
                label="Centroid",
            )
            # Display mixed Gaussian distributions
            ax.plot_wireframe(X1, X2, pos)
            ax.set(
                xlabel="x",
                ylabel="y",
                zlabel="Probability density",
                title="Probability density of {}(K = {})".format(
                    self.filename[:5], self.K
                ),
            )
            ax.grid()
            ax.legend()

            def rotate(angle):
                ax.view_init(azim=angle * 5)

            rot_animation = animation.FuncAnimation(
                fig2, rotate, frames=50, interval=200
            )
            rot_animation.save(
                "result/{}_gmm_3d_K{}.gif".format(self.filename[:5], self.K),
                writer="pillow",
                dpi=100,
            )


def main():
    """Perform fitting of data using GMM."""
    parser = argparse.ArgumentParser(prog="main.py", description="GMM")

    parser.add_argument("-f", "--filename", help="File name")
    parser.add_argument("-k", "--k", help="Number of cluster", type=int)

    args = parser.parse_args()

    filename = args.filename
    k = args.k

    path = os.getcwd() + "/../" + filename
    data = np.loadtxt(path, delimiter=",")

    gmm = GMM(data, k, filename)
    gmm.fit()


if __name__ == "__main__":
    main()
