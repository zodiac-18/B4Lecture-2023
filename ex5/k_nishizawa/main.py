"""Perform clustering and MFCC analysis."""
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import k_means
import mfcc_analysis as mfana


def run_k_means(filename):
    """
    Run k_means and plot graph

    Args:
        filename (str): csv file name
    """
    data = pd.read_csv(filename)
    data = np.array(data)
    dimension = len(data[0])

    N = 4

    if dimension == 2:
        fig, ax = plt.subplots((N + 1) // 2, 2, figsize=(12, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        for k in range(0, N):
            x, y, cluster_n = k_means.k_means_2d(data, k + 2)
            ax[k // 2][k % 2].set_title(f"cluster-{k + 2}")
            ax[k // 2][k % 2].scatter(x, y, c=cluster_n, s=30)
            ax[k // 2][k % 2].set_xlabel("x")
            ax[k // 2][k % 2].set_ylabel("y")
        # extract the png so that the filename of the gif is the original filename.
        plt.savefig(f"{filename[2:-4]}.png")

    elif dimension == 3:
        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.2)

        def rotate(angle):
            # rotate 3D graph
            for i in range(N):
                axes[i].view_init(azim=angle)

        axes = []
        for k in range(0, N):
            x, y, z, cluster_n = k_means.k_means_3d(data, k + 2)
            ax = fig.add_subplot((N + 1) // 2, 2, k + 1, projection="3d")
            ax.scatter(x, y, z, c=cluster_n, s=10)
            ax.set_title(f"cluster-{k + 2}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            axes.append(ax)
        rot_animation = animation.FuncAnimation(fig, rotate, frames=360, interval=30)
        # extract the gif so that the filename of the gif is the original filename.
        rot_animation.save(f"{filename[2:-4]}.gif", writer="pillow", dpi=100)


def main():
    """
    Perform k_mean and mfcc
    """
    parser = argparse.ArgumentParser(
        description="Perform clustering and MFCC analysis."
    )

    parser.add_argument("-m", "--mode", help="k: k-means, m: mfcc")
    parser.add_argument("-f", "--filename", help="File name")

    args = parser.parse_args()

    mode = args.mode
    filename = args.filename

    if mode == "k":
        run_k_means(filename)

    elif mode == "m":
        mfana.mfcc_plot(filename)


if __name__ == "__main__":
    main()
