"""To principal component analysis."""
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np

import csvOpe
import PCA

if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="This is a program to principal component analysis."
    )
    parser.add_argument("path", help="dataset path")

    # read out args
    args = parser.parse_args()

    # read out data
    data = csvOpe.read_csv(args.path)

    # create save image name
    i_name = re.sub(r".+\\", "", args.path)
    i_name = re.sub(r"\..+", "", i_name)

    # normalization
    data_n = PCA.normalize(data)

    # compute eigen values and vectors
    pairs, c_rate = PCA.PCA(data_n)

    # 2-dim
    if len(data_n[0]) == 2:
        # plot range
        min_d = np.min(data_n.T[0])
        max_d = np.max(data_n.T[0])

        # compute slope
        slopes = pairs[:, 2] / pairs[:, 1]

        # plot new axis
        plt.figure()
        plt.plot(
            data_n.T[0],
            data_n.T[1],
            ".",
        )
        min_r = min_d / pairs[:, 1]
        max_r = max_d / pairs[:, 1]
        plt.plot(
            [min_d, max_d],
            [min_d * slopes[0], max_d * slopes[0]],
            label="c rate=" + "{:.3f}".format(c_rate[0]),
        )
        plt.plot(
            [min_d, max_d],
            [min_d * slopes[1], max_d * slopes[1]],
            label="c rate=" + "{:.3f}".format(c_rate[1]),
        )
        plt.xlabel("$X_{1}$")
        plt.ylabel("$X_{2}$")
        plt.title("graph")
        plt.savefig(i_name + "_PCA")
        plt.legend()

        # dimension compression
        new_data = PCA.dimComp(data_n, pairs, 2)

    # 3-dim
    elif len(data_n[0]) == 3:
        # plot range
        min_d = np.min(data_n.T[0])
        max_d = np.max(data_n.T[0])

        min_r = np.abs(min_d / pairs[:, 1])
        max_r = np.abs(max_d / pairs[:, 1])
        # plot data
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            data_n.T[0], data_n.T[1], data_n.T[2], edgecolors="red", facecolor="None"
        )
        ax.set_xlabel("$X_{1}$")
        ax.set_ylabel("$X_{2}$")
        ax.set_zlabel("$X_{3}$")

        # plot new axis
        ax.plot(
            [-min_r[0] * pairs[0, 1], max_r[0] * pairs[0, 1]],
            [-min_r[0] * pairs[0, 2], max_r[0] * pairs[0, 2]],
            [-min_r[0] * pairs[0, 3], max_r[0] * pairs[0, 3]],
            label="c rate=" + "{:.3f}".format(c_rate[0]),
        )
        ax.plot(
            [-min_r[1] * pairs[1, 1], max_r[1] * pairs[1, 1]],
            [-min_r[1] * pairs[1, 2], max_r[1] * pairs[1, 2]],
            [-min_r[1] * pairs[1, 3], max_r[1] * pairs[1, 3]],
            label="c rate=" + "{:.3f}".format(c_rate[1]),
        )
        ax.plot(
            [-min_r[2] * pairs[2, 1], max_r[2] * pairs[2, 1]],
            [-min_r[2] * pairs[2, 2], max_r[2] * pairs[2, 2]],
            [-min_r[2] * pairs[2, 3], max_r[2] * pairs[2, 3]],
            label="c rate=" + "{:.3f}".format(c_rate[2]),
        )
        ax.legend()

        ax.view_init(elev=30, azim=-60)
        plt.savefig(i_name + "_PCA")
        # dimension compression
        comp_data = PCA.dimComp(data_n, pairs, 2)

        plt.figure()
        plt.plot(comp_data.T[0], comp_data.T[1], ".")
        plt.xlabel("$PC_{1}$" + " $C_{rate}:$ " + "{:.3f}".format(c_rate[0]))
        plt.ylabel("$PC_{2}$" + " $C_{rate}:$ " + "{:.3f}".format(c_rate[1]))
        plt.title("compressed outcome")
        plt.savefig(i_name + "_Compression")

    # 4-dim or more
    else:
        PCA.PlotConRate(c_rate)
        plt.savefig(i_name + "_Cumulative")

    plt.show()
