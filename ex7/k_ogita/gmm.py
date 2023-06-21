#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gaussian Mixture Model."""
import argparse
import collections
import csv
import os
from scipy import stats as st
import numpy.random as rd

import matplotlib.pyplot as plt
import numpy as np


class GMM:
    """Gaussian Mixture Model."""

    def __init__(self, data, epsilon, K=3):
        """
        Initialize instance variable.

        Args:
            cluster_n (int): The number of cluster.
            dim (int, optional): Dimension of the data. Defaults to 2.
        """
        indices = np.random.choice(range(len(data)), size=K, replace=False)
        self.K = K
        self.mu = data[indices]
        self.pi = np.concatenate([np.full(K-1, 1/K), [1 - (K-1)*(1/K)]])
        self.cov = np.array([np.cov(data.T) + np.eye(data.shape[1]) for _ in range(K)])
        self.epsilon = epsilon

    def fit(self, data):
        """
        Fitting to data using Gaussian Mixture Model.

        Args:
            data (ndarray): Input data.
        """
        log_likelihood = self.calc_loglikelihood(data)
        log_likelihood_list = [log_likelihood]
        while True:
            self._em_step(data)
            log_likelihood = self.calc_loglikelihood(data)
            log_likelihood_list.append(log_likelihood)
            if np.abs(log_likelihood_list[-1] - log_likelihood_list[-2]) < self.epsilon:
                break
        return log_likelihood_list
        
    def _em_step(self, data):
        # ============== E step ==============
        likelihood = self.calc_likelihood(data)
        # Calculate (𝜋ₖ * 𝑁(xₙ|𝝁ₖ,𝚺ₖ)) / (∑[𝑗=1 to 𝐾](𝜋ⱼ * 𝑁(xₙ|𝝁ⱼ,𝚺ⱼ)))
        gamma = (likelihood.T / np.sum(likelihood, axis=1))

        # ============== M step ==============
        N_k = np.sum(gamma, axis=1)
        self.mu = (gamma @ data) / N_k[:, np.newaxis]
        self.pi = N_k / np.sum(N_k)
        diff = data - self.mu[:, np.newaxis, :]
        for k in range(len(N_k)):
            self.cov[k] = gamma[k] * diff[k].T @ diff[k] / N_k[k]

    def _calc_gauss_pdf(self, data, mu, cov):
        n_features = data.shape[0]
        # Calculate N(xₙ|𝝁ₖ,𝚺ₖ)
        gauss_pdf = 1 / (np.power(np.sqrt(2 * np.pi), n_features) * np.sqrt(np.linalg.det(cov))) * np.exp(-0.5 * (data - mu) @ np.linalg.inv(cov) @ (data - mu).T)
        return gauss_pdf

    def calc_likelihood(self, data):
        likelihood = np.zeros((data.shape[0], self.K))
        for k in range(self.K):
            # Calculate 𝜋ₖ * N(xₙ|𝝁ₖ,𝚺ₖ)
            likelihood[:, k] = [self.pi[k] * self._calc_gauss_pdf(x, self.mu[k], self.cov[k]) for x in data]
        return likelihood
    
    def calc_loglikelihood(self, data):
        likelihood = self.calc_likelihood(data)
        return np.sum(np.log(np.sum(likelihood, axis=1)), axis=0)


def main():
    """Classify the data using k-means method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to the input data")
    parser.add_argument(
        "-k", help="The number of cluster", type=int, default=3
    )
    parser.add_argument(
        "-e", "--epsilon", help="Epsilon", type=float, default=0.00001
    )
    parser.add_argument(
        "-save", "--save_fig", help="Whether to save figure or not", action="store_true"
    )
    args = parser.parse_args()

    # Get the path of data
    path = args.path
    # Get the filename of data from the path
    file_name = os.path.splitext(os.path.basename(path))[0]
    
    epsilon = args.epsilon

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
    
    model = GMM(data, epsilon)
    b = model.fit(data)
    
    """

    model = KMeans(cluster_n, dim)
    count_labels = np.zeros(data.shape[0])
    while not (
        np.all(count_labels > min_cluster_data_num) and len(count_labels) == cluster_n
    ):
        model.fit(data)
        labels = model.labels
        count_labels = np.array(list(collections.Counter(labels).values()))
    scatter_label_list = [f"cluster_{k:0=2}" for k in range(cluster_n)]

    fig = plt.figure(figsize=(15, 10))

    # k-means clustering　for two-dimensional data
    if dim == 2:
        ax = fig.add_subplot(111)
        for i in range(cluster_n):
            indices = np.where(labels == i)[0]
            ax.scatter(
                data[indices, 0],
                data[indices, 1],
                color=cmap((i + 1) / cluster_n),
                label=scatter_label_list[i],
            )
    # k-means clustering　for three-dimensional data
    elif dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        for i in range(cluster_n):
            indices = np.where(labels == i)[0]
            ax.scatter3D(
                data[indices, 0],
                data[indices, 1],
                data[indices, 2],
                color=cmap((i + 1) / cluster_n),
                label=scatter_label_list[i],
            )
        ax.set_zlabel("$z$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"$k$-means clustering of {file_name} ($k={cluster_n}$)")
    ax.legend(loc="upper left")
    if args.save_fig:
        fig.savefig(f"fig/{file_name}_{cluster_n}.png")
    plt.show()
    """


if __name__ == "__main__":
    main()
