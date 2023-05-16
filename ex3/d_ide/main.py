import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="data csv file",
    )

    return parser.parse_args()


def open_csv(file_path):
    data_set = np.loadtxt(
        fname=file_path,
        dtype="str",
        delimiter=",",
    )

    data_set = np.delete(data_set, 0, 0)

    data_set = data_set.astype(np.float16)

    return data_set

def scat_plot(data_set):
    if data_set.shape[1] == 2:
        x = data_set[:, 0]
        y = data_set[:, 1]

        plt.scatter(x, y)
        plt.title("Original Scatter Plot")
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")

        plt.show()
    elif data_set.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = data_set[:, 0]
        y = data_set[:, 1]
        z = data_set[:, 2]

        ax.scatter(x, y, z)

        ax.set_title("Original Scatter Plot")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        plt.show()


def main():
    args = parse_args()

    file_path = args.csv_file

    data_list = open_csv(file_path)

    scat_plot(data_list)


if __name__ == "__main__":
    main()