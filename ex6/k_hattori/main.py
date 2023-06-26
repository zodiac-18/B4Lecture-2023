"""Read files and perform PCA."""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """
    Load csv files.

    Args:
        path (str): A file path.

    Returns:
        ndarray: Contents of csv file.
    """
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        buf = [row for row in reader]
    # Convert to ndarray in float64
    array = np.array(buf[1:])
    array = array.astype(np.float64)

    return array


def PCA(data):
    """
    Perform PCA to find contribution rate and projection matrix.

    Args:
        data (ndarray): Target array.

    Returns:
        ndarray: Contriburtion rate array.
        ndarray: Projection matrix.
    """
    dimension = data.shape[1]
    std_data = np.zeros(data.shape)
    # Standardization of data
    for i in range(dimension):
        # Means and standard deviations
        mu = np.mean(data[:, i])
        sigma = np.std(data[:, i])
        std_data[:, i] = (data[:, i] - mu) / sigma
    # Calculate covariance
    cov = np.cov(np.array([std_data[:, i] for i in range(dimension)]))
    # Eigenvalues and eigenvectors
    eig_value, eig_vector = np.linalg.eig(cov)
    sort_eig = np.sort(eig_value)[::-1]
    j = 0
    sort_vector = np.zeros(eig_vector.shape)
    for i in sort_eig:
        sort_vector[:, j] = np.ravel(eig_vector[:, eig_value == i])
        j += 1
    # Calculation of Contribution Ratio
    cont = sort_eig / np.sum(sort_eig)

    return cont, sort_vector


def main():
    """
    Perform PCA and plot data.

    Returns:
        None
    """
    # make parser
    parser = argparse.ArgumentParser(
        prog="main.py",
        usage="Demonstration of argparser",
        description="description",
        epilog="end",
        add_help=True,
    )
    # add arguments
    parser.add_argument("-f", dest="filename", help="Filename", required=True)
    # parse arguments
    args = parser.parse_args()

    path = args.filename
    data = load_csv(path)
    dimension = data.shape[1]

    # Principal component analysis
    Cont_rate, W = PCA(data)
    print("--------contribution rate--------")
    print(Cont_rate)
    print("--------projective matrix--------")
    print(W)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    if dimension == 2:
        str1 = f"Contribution rate:{Cont_rate[0]:.3f}"
        str2 = f"Contribution rate:{Cont_rate[1]:.3f}"
        plt.scatter(
            data[:, 0],
            data[:, 1],
            edgecolors="magenta",
            facecolor="None",
            label="data1",
        )
        plt.axline((0, 0), W[:, 0], color="blue", label=str1)
        plt.axline((0, 0), W[:, 1], color="red", label=str2)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    elif dimension == 3:
        x = np.linspace(np.min(data[:, 0]), np.max(data[:, 1]), 100)
        a_yx = W[1] / W[0]
        a_zx = W[2] / W[0]
        # Dimensional reduction
        data_2d = np.dot(data, W[:, :2])

        str1 = f"Contribution rate:{Cont_rate[0]:.3f}"
        str2 = f"Contribution rate:{Cont_rate[1]:.3f}"
        str3 = f"Contribution rate:{Cont_rate[2]:.3f}"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            edgecolors="magenta",
            facecolor="None",
            label="data2",
        )
        ax.plot(x, a_yx[0] * x, a_zx[0] * x, color="blue", label=str1)
        ax.plot(x, a_yx[1] * x, a_zx[1] * x, color="red", label=str2)
        ax.plot(x, a_yx[2] * x, a_zx[2] * x, color="green", label=str3)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        plt.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.show()

        plt.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            edgecolor="magenta",
            facecolor="None",
            label="data2 in 2D",
        )
        plt.xlim(-3, 3)
        plt.ylim(-1, 1)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    else:
        # Total contribution rate
        total_cont = np.array(
            [np.sum(Cont_rate[:i]) for i in range(len(Cont_rate))]
            )
        print("Order:", np.min(np.where(total_cont >= 0.9)))
        plt.plot(total_cont, color="blue", label="Total contribution rate")
        plt.axline((0, 0.9), (1, 0.9), color="red", label="Threshold 90%")
        plt.xlabel("Number of principal components")
        plt.ylabel("Contribution rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
    exit(1)
