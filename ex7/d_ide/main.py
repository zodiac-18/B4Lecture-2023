'Main.'
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gmm


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="Perform pca")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="data1.csv",
        help="data csv file",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=2,
        help="number of cluster",
    )

    return parser.parse_args()


def open_csv(file_path):
    """Read csv file.
    Args:
        file_path (str): Csv file to read
    Returns:
        ndarray: Data read
    """
    df = pd.read_csv(file_path)
    data = df.values
    return df, data


def plot2d(data):
    """Plot 2 dimension data.
    Args:
        data (ndarray): target data
    """

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], label="original data")
    ax.set_title("Observed data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()
    plt.close()


def plot1d(data):
    """Plot 3 dimension data
    Args:
        data (ndarray): target data
    """
    y = np.zeros_like(data)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(data, y, label="original data", s=5)
    ax.set_title("Observed data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    #plt.savefig("result/pca_data2.png")
    plt.show()
    plt.close()


def main():
    """Calculate GMM main file.
    """
    args = parse_args()
    fname = args.csv_file
    cluster = args.cluster

    df, data = open_csv(fname)
    num, dim = data.shape
    eps = 0.01
    vec0, cov0, pi0 = gmm.ini(data, cluster)
    count, gamma, vec, cov, pi, log_list = gmm.EM(data, vec0, cov0, pi0, eps)

    clu = np.argmax(gamma, axis=0)
    if dim == 1:
        gmm.scatter_1d(data, clu, vec, cov, pi)
    elif dim == 2:
        gmm.scatter_2d(data, clu, vec, cov, pi)
    else:
        "error: dimension"
    gmm.logplot(log_list)


if __name__ == "__main__":
    main()
