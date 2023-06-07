"""Princple Component Analysis."""
import matplotlib.pyplot as plt
import numpy as np


class PCA:
    """PCA module."""

    def __init__(self, n_componets):
        self.n_components = n_componets

    def fit(self, data):
        self.data = self._standardize(data)
        cov = np.cov(self.data, rowvar=False)
        self.eigen_value, self.eigen_vector = np.linalg.eig(cov)
        self.W = self.eigen_vector[:, np.argsort(-self.eigen_value)][
            :, : self.n_components
        ]

    def transform(self, data):
        return data @ self.W

    def eigen(self):
        return self.eigen_value, self.eigen_vector

    def contribution_rate(self):
        return self.eigen_value / np.sum(self.eigen_value)

    def _standardize(self, data):
        data = (data - data.mean(axis=0)) 
        data = data / np.std(data, axis=0)
        return data


def data1():
    data = np.loadtxt("dataset/data1.csv", delimiter=",")
    pca = PCA(n_componets=2)
    pca.fit(data)
    eigen_value, eigen_vector = pca.eigen()
    contribution_rate = pca.contribution_rate()
    slope = eigen_vector[1] / eigen_vector[0]
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 1]), 100)

    # draw oroginal data
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], label="original data")
    ax.plot(x, slope[0] * x, label=f'princple component 1(c rate={contribution_rate[0]:.3f})')
    ax.plot(x, slope[1] * x, label=f'princple component 2(c rate={contribution_rate[1]:.3f})')
    ax.set_title("Observed data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.savefig("pca_data1.png")


def data2():
    data = np.loadtxt("dataset/data2.csv", delimiter=",")
    pca = PCA(n_componets=2)

    pca.fit(data)
    transformed_data = pca.transform(data)
    eigen_value, eigen_vector = pca.eigen()
    contribution_rate = pca.contribution_rate()
    slope_xy = eigen_vector[1] / eigen_vector[0]
    slope_xz = eigen_vector[2] / eigen_vector[0]
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 1]), 100)

    # draw oroginal data
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], label="original data")
    ax.plot(x, slope_xy[0] * x, slope_xz[0] * x, label=f'princple component 1(c rate={contribution_rate[0]:.3f})')
    ax.plot(x, slope_xy[1] * x, slope_xz[1] * x, label=f'princple component 2(c rate={contribution_rate[1]:.3f})')
    ax.plot(x, slope_xy[2] * x, slope_xz[2] * x, label=f'princple component 3(c rate={contribution_rate[2]:.3f})')
    ax.set_title("Observed data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.savefig("pca_data2_origin.png")
    plt.clf()

    # draw transformed data
    ax = fig.add_subplot(111)
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], label="transformed data")
    ax.set_title("transformed data")
    ax.set_xlabel(f'princple component 1(c rate={contribution_rate[0]:.3f})')
    ax.set_ylabel(f'princple component 2(c rate={contribution_rate[1]:.3f})')
    ax.legend()

    # plt.show()
    plt.savefig("pca_data2_compress.png")


def data3():
    data = np.loadtxt("dataset/data3.csv", delimiter=",")
    pca = PCA(n_componets=None)
    pca.fit(data)
    contribution_rate = pca.contribution_rate()
    for i in range(1, 100):
        total_rate = np.sum(contribution_rate[:i])
        if 0.85 < total_rate < 0.92: 
            print(i, f'{total_rate=}')


if __name__ == "__main__":
    # data1()
    data2()
    # data3()
