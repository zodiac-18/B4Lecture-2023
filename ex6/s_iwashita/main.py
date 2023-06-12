"""Perform principal component analysis."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class PCA:
    """This is a class associated with PCA."""

    def __init__(self, filename):
        """Define arguments."""
        self.filename = filename

    def fit(self):
        """Perform PCA."""
        self.data = np.loadtxt(self.filename, delimiter=",")
        # Standardize data
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
        """Transform data."""
        self.data_tf = np.dot(self.data, self.w)

    def plot_scatter(self):
        """Generate a scatter plot."""
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
                ax.scatter(
                    self.data[:, 0],
                    self.data[:, 1],
                    c="None",
                    edgecolor="magenta",
                    label="{}".format(self.filename[3:8]),
                )
                ax.plot(
                    x,
                    x * slope_xy[0],
                    c="b",
                    label="Contribution rate: {:.3f}".format(self.contribution_rate[0]),
                )
                ax.plot(
                    x,
                    x * slope_xy[1],
                    c="g",
                    label="Contribution rate: {:.3f}".format(self.contribution_rate[1]),
                )
                ax.set(
                    title="{}".format(self.filename[3:8]),
                    xlabel="$x_1$",
                    ylabel="$x_2$",
                    xlim=(x_min, x_max),
                    ylim=(y_min, y_max),
                )
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            elif self.n_components == 3:
                z_min = int(np.min(self.data[:, 2]) // 1)
                z_max = int(np.max(self.data[:, 2]) // 1 + 1)
                slope_xz = self.eig_vec[2] / self.eig_vec[0]
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    self.data[:, 0],
                    self.data[:, 1],
                    self.data[:, 2],
                    c="None",
                    edgecolor="magenta",
                    label="{}".format(self.filename[3:8]),
                )
                ax.plot(
                    x,
                    x * slope_xy[0],
                    x * slope_xz[0],
                    c="b",
                    label="Contribution rate: {:.3f}".format(self.contribution_rate[0]),
                )
                ax.plot(
                    x,
                    x * slope_xy[1],
                    x * slope_xz[1],
                    c="g",
                    label="Contribution rate: {:.3f}".format(self.contribution_rate[1]),
                )
                ax.plot(
                    x,
                    x * slope_xy[2],
                    x * slope_xz[2],
                    c="r",
                    label="Contribution rate: {:.3f}".format(self.contribution_rate[2]),
                )
                ax.set(
                    title="{}".format(self.filename[3:8]),
                    xlabel="$x_1$",
                    ylabel="$x_2$",
                    zlabel="$x_3$",
                    xlim=(x_min, x_max),
                    ylim=(y_min, y_max),
                    zlim=(z_min, z_max),
                )
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.zaxis.set_major_locator(MaxNLocator(integer=True))
                def rotate(angle):
                    ax.view_init(azim=angle * 5)
                ax.grid(ls="--")
                ax.legend(loc="upper left")
                rot_animation = animation.FuncAnimation(fig, rotate, frames=50, interval=200)
                rot_animation.save("result/{}.gif".format(self.filename[3:8]), writer="pillow", dpi=100)
            ax.grid(ls="--")
            ax.legend(loc="upper left")
            plt.savefig("result/{}.png".format(self.filename[3:8]))
            plt.close()

    def plot_cumulative_contribution_rate(self):
        """Calculate and plot cumulative contribution rate."""
        self.cum_con_rate = np.cumsum(self.contribution_rate)
        self.dimension_above_90 = np.min(np.where(self.cum_con_rate >= 0.9)) + 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(self.cum_con_rate) + 1), self.cum_con_rate, c="b", label="Cumulative contribution rate")
        ax.plot(
            self.dimension_above_90, self.cum_con_rate[self.dimension_above_90 - 1], "D-", c="g"
        )
        ax.axline(
            (self.dimension_above_90, 0),
            (self.dimension_above_90, self.cum_con_rate[self.dimension_above_90]),
            c="g",
            label="{}".format(self.dimension_above_90),
        )
        ax.set(
            title="{}".format(self.filename[3:8]),
            xlabel="Dimension",
            ylabel="Cumulative contribution rate",
            yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axhline(y=0.9, c="r", label="Line of 90%")
        ax.grid()
        ax.legend(loc="lower right")
        plt.savefig("result/{}_cum_con.png".format(self.filename[3:8]))
        plt.close()

    def dimensional_compression(self):
        """Perform dimensional compression."""
        if self.dimension_above_90 == 2 or self.dimension_above_90 == 3:
            x_min = int(np.min(self.data_tf[:, 0]) // 1)
            x_max = int(np.max(self.data_tf[:, 0]) // 1 + 1)
            y_min = int(np.min(self.data_tf[:, 1]) // 1)
            y_max = int(np.max(self.data_tf[:, 1]) // 1 + 1)
            fig = plt.figure()
            if self.dimension_above_90 == 2:
                ax = fig.add_subplot(111)
                ax.scatter(
                    self.data_tf[:, 0],
                    self.data_tf[:, 1],
                    c="None",
                    edgecolor="magenta",
                )
                ax.set(
                    title="Comperssed {}".format(self.filename[3:8]),
                    xlabel="$x_1$",
                    ylabel="$x_2$",
                    xlim=(x_min, x_max),
                    ylim=(y_min, y_max),
                )
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            elif self.dimension_above_90 == 3:
                z_min = int(np.min(self.data_tf[:, 2]) // 1)
                z_max = int(np.max(self.data_tf[:, 2]) // 1 + 1)
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    self.data_tf[:, 0],
                    self.data_tf[:, 1],
                    self.data_tf[:, 2],
                    c="None",
                    edgecolor="magenta",
                )
                ax.set(
                    title="Comperssed {}".format(self.filename[3:8]),
                    xlabel="$x_1$",
                    ylabel="$x_2$",
                    zlabel="$x_3$",
                    xlim=(x_min, x_max),
                    ylim=(y_min, y_max),
                    zlim=(z_min, z_max),
                )
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.zaxis.set_major_locator(MaxNLocator(integer=True))
                def rotate(angle):
                    ax.view_init(azim=angle * 5)
                rot_animation = animation.FuncAnimation(fig, rotate, frames=50, interval=200)
                rot_animation.save("result/{}.gif".format(self.filename[3:8]), writer="pillow", dpi=100)
            ax.grid(ls="--")
            plt.savefig("result/{}_compressed.png".format(self.filename[3:8]))
            plt.close()


def main():
    """Perform PCA."""
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
