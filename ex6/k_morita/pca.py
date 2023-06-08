"""Princple Component Analysis."""
from matplotlib.animation import FuncAnimation
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
        self.eigen_vector = self.eigen_vector[:, np.argsort(-self.eigen_value)]
        self.eigen_value = sorted(self.eigen_value, reverse=True)
        self.W = self.eigen_vector[:, : self.n_components]

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

    # draw original data
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    def init():
        ax.scatter(data[:, 0], data[:, 1], label="original data")
        ax.plot(x, slope_xy[0] * x, slope_xz[0] * x, label=f'princple component 1(c rate={contribution_rate[0]:.3f})')
        ax.plot(x, slope_xy[1] * x, slope_xz[1] * x, label=f'princple component 2(c rate={contribution_rate[1]:.3f})')
        ax.plot(x, slope_xy[2] * x, slope_xz[2] * x, label=f'princple component 3(c rate={contribution_rate[2]:.3f})')
        ax.set_title("Observed data")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=30, azim=45, roll=15)
        ax.legend()

    def update(frame):
        if 0 <= frame < 30:
            ax.view_init(elev=30+frame, azim=45, roll=15)
        elif 30 <= frame < 60:
            ax.view_init(elev=30+30, azim=45+frame%30, roll=15)
        else:
            pass
    ani = FuncAnimation(fig, update, frames=np.arange(60), init_func=init, repeat=False)
    ani.save("pca_data2_origin.gif", writer="pillow")
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
    x = np.arange(100)
    total_rate = [np.sum(contribution_rate[:i]) for i in x]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(x+1, total_rate, label="total contribution rate")
    ax.hlines(0.9, xmin=0, xmax=100, colors="orange")
    ax.set_title("n_component - Total Contoribution Rate")
    ax.set_xlabel("component number")
    ax.set_ylabel("total contoribution rate")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.savefig("pca_data3.png")

if __name__ == "__main__":
    # data1()
    # data2()
    data3()
