import argparse

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class PCA:

    def __init__(self, filename):
        self.filename = filename
        self.data_ss = None
        self.n_components = None
        self.standardized = None
        self.eig = None
        self.eig_vec = None
        self.data_tf = None

    def fit(self):
        #
        self.data = np.loadtxt(self.filename, delimiter=",")
        #
        self.data_ss = scipy.stats.zscore(self.data)
        # get n_components
        self.n_components = len(self.data[0])
        # Find the variance-covariance matrix of the data
        self.v_cov = np.cov(self.data_ss.T)
        # Find the eigenvalues and eigenvectors of the variance-covariance matrix
        self.eig, self.eig_vec = np.linalg.eig(self.v_cov)
        # Sort eigenvalue in descending order
        self.eig_vec = self.eig_vec[:, np.argsort(-self.eig)]
        self.eig = sorted(self.eig, reverse=True)
        # Create a projection matrix from eigenvectors
        self.w = self.eig_vec[:, : self.n_components]
        # Calculate contribution rate
        self.contribution_rate = self.eig / np.sum(self.eig)

    def transform(self):
        self.data_tf = np.dot(self.data, self.w)

    def plot_scatter(self):
        if self.n_components == 2 or self.n_components == 3:
            x_min = int(np.min(self.data[:, 0]) // 1)
            x_max = int(np.max(self.data[:, 0]) // 1 + 1)
            x = np.linspace(x_min, x_max, 100)
            y_min = int(np.min(self.data[:, 1]) // 1)
            y_max = int(np.max(self.data[:, 1]) // 1 + 1)
            slope_xy = self.eig_vec[1] / self.eig_vec[0]

            fig = plt.figure()
            if self.n_components == 2:
                ax = fig.add_subplot(111)
                ax.scatter(self.data[:, 0], self.data[:, 1], c="None", edgecolor="magenta", label="{}".format(self.filename[3:8]))
                ax.plot(x, x * slope_xy[0], c="b", label="Contribution rate: {:.3f}".format(self.contribution_rate[0]))
                ax.plot(x, x * slope_xy[1], c="g", label="Contribution rate: {:.3f}".format(self.contribution_rate[1]))
                ax.set(title="{}".format(self.filename[3:8]), xlabel="$x_1$", ylabel="$x_2$", xlim=(x_min, x_max), ylim=(y_min, y_max), xticks=range(x_min, x_max + 1, 1), yticks=range(y_min, y_max + 1, 1))

            elif self.n_components == 3:
                z_min = int(np.min(self.data[:, 2]) // 1)
                z_max = int(np.max(self.data[:, 2]) // 1 + 1)
                slope_xz = self.eig_vec[2] / self.eig_vec[0]
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c="None", edgecolor="magenta", label="{}".format(self.filename[3:8]))
                ax.plot(x, x * slope_xy[0], x * slope_xz[0], c="b", label="Contribution rate: {:.3f}".format(self.contribution_rate[0]))
                ax.plot(x, x * slope_xy[1], x * slope_xz[1], c="g", label="Contribution rate: {:.3f}".format(self.contribution_rate[1]))
                ax.plot(x, x * slope_xy[2], x * slope_xz[2], c="r", label="Contribution rate: {:.3f}".format(self.contribution_rate[2]))
                ax.set(title="{}".format(self.filename[3:8]), xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(z_min, z_max), xticks=range(x_min, x_max + 1, 1), yticks=range(y_min, y_max + 1, 1), zticks=range(z_min, z_max + 1, 1))
            ax.grid(ls="--")
            ax.legend(loc="upper left")
            plt.savefig("result/{}.png".format(self.filename[3:8]))
            plt.close()

    def plot_cumulative_contribution_rate(self):
        self.cum_con_rate = np.cumsum(self.contribution_rate)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.cum_con_rate, c="b", label="Cumulative contribution rate")
        ax.set(title="{}".format(self.filename[3:8]), xlabel="Index of contribution rate", ylabel="Cumulative contribution rate", yticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axhline(y=0.9, c="r", label="Line of 90%")
        ax.grid()
        ax.legend(loc="lower right")
        plt.savefig("result/{}_cum_con.png".format(self.filename[3:8]))
        plt.close()

    def dimensional_compression(self):
        idx_compression = np.min(np.where(self.cum_con_rate >= 0.9)) + 1
        if idx_compression == 2 or idx_compression == 3:
            x_min = int(np.min(self.data_tf[:, 0]) // 1)
            x_max = int(np.max(self.data_tf[:, 0]) // 1 + 1)
            y_min = int(np.min(self.data_tf[:, 1]) // 1)
            y_max = int(np.max(self.data_tf[:, 1]) // 1 + 1)
            fig = plt.figure()
            if idx_compression == 2:
                ax = fig.add_subplot(111)
                ax.scatter(self.data_tf[:, 0], self.data_tf[:, 1], c="None", edgecolor="magenta")
                ax.set(title="Comperssed {}".format(self.filename[3:8]), xlabel="$x_1$", ylabel="$x_2$", xlim=(x_min, x_max), ylim=(y_min, y_max), xticks=range(x_min, x_max + 1, 1), yticks=range(y_min, y_max + 1, 1))
            elif idx_compression == 3:
                z_min = int(np.min(self.data_tf[:, 2]) // 1)
                z_max = int(np.max(self.data_tf[:, 2]) // 1 + 1)
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(self.data_tf[:, 0], self.data_tf[:, 1], self.data_tf[:, 2], c="None", edgecolor="magenta")
                ax.set(title="Comperssed {}".format(self.filename[3:8]), xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(z_min, z_max), xticks=range(x_min, x_max + 1, 1), yticks=range(y_min, y_max + 1, 1), zticks=range(z_min, z_max + 1, 1))
            ax.grid(ls="--")
            plt.savefig("result/{}_compressed.png".format(self.filename[3:8]))
            plt.close


"""
d=2:
    scatter
    plot

d=3:
    scatter 
    plot
    compress
    contribution rate

d>4:
    contribution rate
    n_compress <= 3:
        compress
"""


def main():
    parser = argparse.ArgumentParser(description="This program performs PCA.")

    parser.add_argument("-f", "--filename", help="FIle name for PCA")

    args = parser.parse_args()
    filename = args.filename

    # Principal component analysis
    pca = PCA(filename)
    pca.fit()
    pca.transform()
    pca.plot_scatter()
    pca.plot_cumulative_contribution_rate()
    pca.dimensional_compression()

if __name__ == "__main__":
    main()