import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihoods = None

    def __initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.weights = np.full(self.n_components, 1.0 / self.n_components)
        random_indices = np.random.choice(range(n_samples), size=self.n_components, replace=False)
        self.means = X[random_indices]
        self.covariances = [np.cov(X.T) + np.eye(n_features) for _ in range(self.n_components)]
        self.log_likelihoods = np.array([])

    def __e_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self.__calculate_likelihood(X, self.means[k], self.covariances[k])
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities

    def __calculate_likelihood(self, X, mean, covariance):
        n_features = X.shape[1]
        constant = 1.0 / ((2 * np.pi) ** (n_features / 2) * np.sqrt(np.linalg.det(covariance)))
        print(covariance.shape)
        exponent = -0.5 * np.sum(np.dot((X - mean), np.linalg.inv(covariance)) * (X - mean), axis=1)
        return constant * np.exp(exponent)

    def __m_step(self, X, responsibilities):
        total_responsibilities = np.sum(responsibilities, axis=0)
        self.weights = total_responsibilities / X.shape[0]
        self.means = np.dot(responsibilities.T, X) / total_responsibilities[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot((diff * responsibilities[:, k][:, np.newaxis]).T, diff) / total_responsibilities[k]

    def __is_finish_em(self, old_parameters):
        if old_parameters is None:
            return False
        old_weights, old_means, old_covariances = old_parameters
        weights_diff = np.max(np.abs(self.weights - old_weights))
        means_diff = np.max(np.abs(self.means - old_means))
        covariances_diff = np.max([np.max(np.abs(self.covariances[k] - old_covariances[k])) for k in range(self.n_components)])
        return weights_diff < self.tol and means_diff < self.tol and covariances_diff < self.tol

    def calculate_log_likelihood(self, X):
        weight_likelihood = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            weight_likelihood[:, k] = self.weights[k] * self.__calculate_likelihood(X, self.means[k], self.covariances[k])
        log_likelihood = np.log(np.sum(weight_likelihood, axis=1))
        return np.sum(log_likelihood)

    def fit(self, X):
        self._GMM__initialize_parameters(X)
        old_parameters = None
        for _ in range(self.max_iter):
            responsibilities = self.__e_step(X)
            self.__m_step(X, responsibilities)
            if self.__is_finish_em(old_parameters):
                break
            old_parameters = (self.weights.copy(), self.means.copy(), self.covariances.copy())
            log_likelihood = self.calculate_log_likelihood(X)
            self.log_likelihoods = np.append(self.log_likelihoods, log_likelihood)

    def plot_log_likelihood(self):
        plt.plot(range(len(self.log_likelihoods)), self.log_likelihoods)
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.title('Log Likelihood vs. Iteration')
        plt.show()

    def plot_mixture_distribution(self, X):
        if X.shape[1] == 1:
            self.plot_1d_mixture_distribution(X)
        elif X.shape[1] == 2:
            self.plot_2d_mixture_distribution(X)
        else:
            print("Unsupported dimensionality for plotting mixture distribution.")

    def __plot_1d_mixture_distribution(self, X):
        x = np.linspace(np.min(X), np.max(X), 1000)
        y = np.zeros_like(x)
        for k in range(self.n_components):
            y += (self.weights[k] * norm.pdf(x, self.means[k], np.sqrt(self.covariances[k]))).flatten()
        plt.plot(x, y)
        # plt.hist(X, bins='auto', density=True)
        plt.scatter(X, np.zeros_like(X), marker="o", s=3)
        plt.scatter(self.means, np.zeros_like(self.means), marker="x", s=20, c="red")
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.title('Mixture Distribution (1D)')
        plt.show()

    def __plot_2d_mixture_distribution(self, X):
        x, y = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100), np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100))
        pos = np.dstack((x, y))
        z = np.zeros_like(x)
        for k in range(self.n_components):
            z += self.weights[k] * multivariate_normal.pdf(pos, mean=self.means[k], cov=self.covariances[k])

        plt.contour(x, y, z)
        plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', s=3)
        plt.scatter(self.means[:, 0], self.means[:, 1], marker="x", s=30, c="red")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Mixture Distribution (2D)')
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    fname = sys.argv[1]
    k = int(sys.argv[2])

    data = np.loadtxt(fname, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    gmm = GMM(n_components=k)
    gmm.fit(data)
    gmm.plot_log_likelihood()
    gmm.plot_mixture_distribution(data)
