"""csv operation."""
import csv

import matplotlib.pyplot as plt
import numpy as np


def plot(data: np.ndarray, centroid: np.ndarray, label: np.ndarray, g_num):
    """To plot data

    Args:
        data (np.ndarray): dataset for plot
        centroid(np.ndarray): centroid
        label (np.ndarray): dataset label
        g_num (_type_): number of groups
    """

    # create graph
    plt.figure()
    colorlist = ["r", "g", "b", "c", "m", "y", "w"]
    centroid = centroid.T

    if len(data) == 2:
        # create graph and plot
        for i in range(g_num):
            plt.plot(
                data[0, label == i],
                data[1, label == i],
                ".",
                label="$G_{" + str(i) + "}$",
                color=colorlist[i],
            )
        plt.plot(
            centroid[0], centroid[1], "p",
            label="centroid", color="k", markersize=10
        )
        plt.xlabel("$X_{1}$")
        plt.ylabel("$X_{2}$")
        plt.title(str(g_num) + " groups clustering")
        plt.legend()

    elif len(data) == 3:
        # create graph and plot
        ax = plt.subplot(projection="3d")
        for i in range(g_num):
            ax.plot(
                data[0, label == i],
                data[1, label == i],
                data[2, label == i],
                ".",
                color=colorlist[i],
                label="$G_{" + str(i) + "}$",
            )
        plt.plot(
            centroid[0],
            centroid[1],
            centroid[2],
            "p",
            label="centroid",
            color="k",
            markersize=10,
        )
        ax.set_xlabel("$X_{1}$")
        ax.set_ylabel("$X_{2}$")
        ax.set_zlabel("$X_{3}$")
        ax.legend()

    else:
        raise ValueError(
            "data dimension should in 2 or 3 but " + str(len(data)))


def read_csv(path: str) -> np.ndarray:
    """read out csv to matrix

    Args:
        path (str): the csv file path

    Returns:
        np.ndarray: data matrix
    """
    # read scv file and change type
    data_array = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            sub_group = []
            for value in line:
                sub_group.append(float(value))
            data_array.append(sub_group)

    # group by dimension
    data_array = np.array(data_array)
    data = data_array.T

    return data


if __name__ == "__main__":
    pass
