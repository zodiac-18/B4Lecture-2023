#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gaussian Mixture Model."""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation as animation


class GMM:
    """Gaussian Mixture Model."""

    def __init__(self, data, epsilon, K=3):
        """
        Initialize instance variable.

        Args:
            data (ndarray): Input data.
            epsilon (float): Threshold for terminating fitting.
            K (int, optional): Number of clusters.
        """
        indices = np.random.choice(range(len(data)), size=K, replace=False)
        self.K = K
        self.mu = data[indices]
        self.pi = np.concatenate([np.full(K - 1, 1 / K), [1 - (K - 1) * (1 / K)]])
        self.cov = np.array([np.cov(data.T) + np.eye(data.shape[1]) for _ in range(K)])
        self.epsilon = epsilon
        self.likelihood = None

    def fit(self, data):
        """
        Fitting to data using Gaussian Mixture Model.

        Args:
            data (ndarray): Input data.
        """
        log_likelihood = self._calc_loglikelihood(data)
        log_likelihood_list = [log_likelihood]
        while True:
            self._em_step(data)
            log_likelihood = self._calc_loglikelihood(data)
            log_likelihood_list.append(log_likelihood)
            if np.abs(log_likelihood_list[-1] - log_likelihood_list[-2]) < self.epsilon:
                break
        return log_likelihood_list

    def _em_step(self, data):
        """
        Run EM algorithm.

        Args:
            data (ndarray): Input data.
        """
        # ============== E step ==============
        likelihood = self.calc_likelihood(data)
        # Calculate (𝜋ₖ * 𝑁(xₙ|𝝁ₖ,𝚺ₖ)) / (∑[𝑗=1 to 𝐾](𝜋ⱼ * 𝑁(xₙ|𝝁ⱼ,𝚺ⱼ)))
        gamma = likelihood.T / np.sum(likelihood, axis=1)

        # ============== M step ==============
        N_k = np.sum(gamma, axis=1)
        self.mu = (gamma @ data) / N_k[:, np.newaxis]
        self.pi = N_k / np.sum(N_k)
        diff = data - self.mu[:, np.newaxis, :]
        for k in range(len(N_k)):
            self.cov[k] = gamma[k] * diff[k].T @ diff[k] / N_k[k]

    def _calc_gauss_pdf(self, data, mu, cov):
        """
        Calculate the probability density function of a Gaussian distribution.

        Args:
            data (_type_): _description_
            mu (_type_): _description_
            cov (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_features = data.shape[0]
        # Calculate N(xₙ|𝝁ₖ,𝚺ₖ)
        gauss_pdf = (
            1
            / (np.power(np.sqrt(2 * np.pi), n_features) * np.sqrt(np.linalg.det(cov)))
            * np.exp(-0.5 * (data - mu) @ np.linalg.inv(cov) @ (data - mu).T)
        )
        return gauss_pdf

    def calc_likelihood(self, data):
        """
        Calculate likelihood.

        Args:
            data (ndarray): Input data.

        Returns:
            ndarray: Likelihood.
        """
        likelihood = np.zeros((data.shape[0], self.K))
        for k in range(self.K):
            # Calculate 𝜋ₖ * N(xₙ|𝝁ₖ,𝚺ₖ) (pdf of mixture gaussian distribution)
            likelihood[:, k] = [
                self.pi[k] * self._calc_gauss_pdf(x, self.mu[k], self.cov[k])
                for x in data
            ]
        return likelihood

    def _calc_loglikelihood(self, data):
        """
        Calculate loglikelihood.

        Args:
            data (ndarray): Input data.

        Returns:
            ndarray: Loglikelihood.
        """
        likelihood = self.calc_likelihood(data)
        return np.sum(np.log(np.sum(likelihood, axis=1)), axis=0)


def main():
    """Classify the data using gaussian mixture model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to the input data")
    parser.add_argument("-k", help="The number of cluster", type=int, default=3)
    parser.add_argument("-e", "--epsilon", help="Epsilon", type=float, default=0.00001)
    args = parser.parse_args()

    # Get the path of data
    path = args.path
    # Get the filename of data from the path
    file_name = os.path.splitext(os.path.basename(path))[0]

    epsilon = args.epsilon

    k = args.k

    # Read data from the csv file
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(list(map(float, row)))
    data = np.array(data)

    # Get the dimention of the data.
    dim = data.shape[1]

    cmap_keyword = "jet"
    cmap = plt.get_cmap(cmap_keyword)

    model = GMM(data, epsilon, k)
    log_likelihood = model.fit(data)

    # Plot log likelihood
    fig1 = plt.figure(figsize=(15, 10))
    ax1 = fig1.add_subplot(111)
    ax1.plot(log_likelihood, c="red", label="log likelihood")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Log likelihood")
    ax1.set_title(f"Log likelihood of {file_name} (k={k})")
    ax1.legend()

    fig1.savefig(f"loglike_{file_name}_k_{k}.png")

    fig2 = plt.figure(figsize=(15, 10))
    ax2 = fig2.add_subplot(111)
    likelihood = model.calc_likelihood(data)
    cluster_label = np.array([np.argmax(likelihood[n]) for n in range(data.shape[0])])
    scatter_label_list = np.array([f"cluster_{k:0=2}" for k in range(k)])

    if dim == 1:
        # Plot clustered data and pdf of mixture gaussian distribution.
        x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 1000)[:, np.newaxis]
        pdf = np.sum(model.calc_likelihood(x), axis=1)
        for i in range(k):
            indices = np.where(cluster_label == i)[0]
            ax2.scatter(
                data[indices],
                np.zeros(len(indices)),
                color=cmap((i + 1) / (k + 2)),
                label=scatter_label_list[i],
            )
        ax2.scatter(
            model.mu[:, 0],
            np.zeros(model.mu.shape[0]),
            color="red",
            label="Centroid",
            marker="x",
        )
        ax2.plot(x, pdf, label="GMM", color="green")
        ax2.set(
            xlabel="x",
            ylabel="Probability density",
            title=f"Clustered data and pdf of {file_name}(k={k})",
            xlim=(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1),
        )
        ax2.grid()
        fig2.savefig(f"cluster_{file_name}_k_{k}.png")

    else:
        # Plot clustered data and a pdf of mixture gaussian distribution by contour lines.
        for i in range(k):
            indices = np.where(cluster_label == i)[0]
            ax2.scatter(
                data[indices, 0],
                data[indices, 1],
                color=cmap((i + 1) / k),
                label=scatter_label_list[i],
            )
        ax2.scatter(model.mu[:, 0], model.mu[:, 1], color="red", marker="x")
        x1 = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
        x2 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.vstack([X1.flatten(), X2.flatten()])
        pdf = np.sum(model.calc_likelihood(X.T), axis=1)
        pdf = pdf.reshape(X1.shape)
        cset = ax2.contour(X1, X2, pdf, cmap=cmap)
        ax2.clabel(cset, fmt="%1.2f", fontsize=9)
        ax2.legend()
        ax2.set(
            xlabel="x",
            ylabel="y",
            xlim=(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1),
            ylim=(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1),
            title=f"Contour map of {file_name} (K = {k})",
        )

        fig2.savefig(f"cluster_{file_name}_k_{k}.png")

        # Plot pdf of mixture Gaussian distribution in 3D.
        fig3 = plt.figure(figsize=(15, 10))
        ax3 = fig3.add_subplot(111, projection="3d")

        def animation_init():
            for i in range(k):
                indices = np.where(cluster_label == i)[0]
                ax3.scatter(
                    data[indices, 0],
                    data[indices, 1],
                    np.zeros(len(indices)),
                    color=cmap((i + 1) / k),
                    label=scatter_label_list[i],
                )
            ax3.scatter(
                model.mu[:, 0],
                model.mu[:, 1],
                np.zeros(model.mu.shape[0]),
                color="red",
                label="Centroid",
                marker="x",
                s=100,
            )
            ax3.plot_surface(
                X1, X2, pdf, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.6
            )
            return (fig3,)

        rotate_azim = np.linspace(60, 420, 120)

        def animate(i):
            ax3.view_init(azim=rotate_azim[i], roll=5)
            return (fig3,)

        ani = animation(
            fig3,
            func=animate,
            init_func=animation_init,
            frames=np.arange(60),
            interval=100,
            blit=True,
            repeat=False,
        )
        ax3.set(
            xlabel="x1",
            ylabel="x2",
            zlabel="Probability density",
            title=f"Probability density of {file_name} (K = {k})",
        )
        ax3.grid()
        ani.save(f"prob_{file_name}_k_{k}.gif", writer="pillow")
    plt.show()


if __name__ == "__main__":
    main()
